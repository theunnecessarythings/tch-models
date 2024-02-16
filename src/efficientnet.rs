use tch::{
    nn::{self, batch_norm2d, conv2d, linear, BatchNormConfig, ConvConfig},
    Tensor,
};

fn make_divisible(v: f64, divisor: i64, min_value: Option<i64>) -> i64 {
    let min_value = min_value.unwrap_or(divisor);
    let new_v = i64::max(
        min_value,
        (v + divisor as f64 / 2.0) as i64 / divisor * divisor,
    );
    if new_v < (0.9 * v) as i64 {
        new_v + divisor
    } else {
        new_v
    }
}

#[derive(Debug, Clone, Copy)]
enum BlockType {
    MBConv,
    FusedMBConv,
}

#[derive(Debug, Clone, Copy)]
struct MBConvConfig {
    expand_ratio: f64,
    ksize: i64,
    stride: i64,
    c_in: i64,
    c_out: i64,
    num_layers: i64,
    block_type: BlockType,
}

impl MBConvConfig {
    fn adjust_channels(c_in: i64, width_mult: f64, min_value: Option<i64>) -> i64 {
        make_divisible(c_in as f64 * width_mult, 8, min_value)
    }

    fn adjust_depth(num_layers: i64, depth_mult: f64) -> i64 {
        (num_layers as f64 * depth_mult).ceil() as i64
    }

    fn mbconv(
        expand_ratio: f64,
        ksize: i64,
        stride: i64,
        c_in: i64,
        c_out: i64,
        num_layers: i64,
        width_mult: f64,
        depth_mult: f64,
    ) -> Self {
        let c_in = Self::adjust_channels(c_in, width_mult, None);
        let c_out = Self::adjust_channels(c_out, width_mult, None);
        let num_layers = Self::adjust_depth(num_layers, depth_mult);
        Self {
            expand_ratio,
            ksize,
            stride,
            c_in,
            c_out,
            num_layers,
            block_type: BlockType::MBConv,
        }
    }

    fn fused_mbconv(
        expand_ratio: f64,
        ksize: i64,
        stride: i64,
        c_in: i64,
        c_out: i64,
        num_layers: i64,
    ) -> Self {
        Self {
            expand_ratio,
            ksize,
            stride,
            c_in,
            c_out,
            num_layers,
            block_type: BlockType::FusedMBConv,
        }
    }
}

fn conv_norm_activation(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    ksize: i64,
    stride: i64,
    groups: i64,
    activation: bool,
) -> impl nn::ModuleT {
    let padding = (ksize - 1) / 2;
    let conv = conv2d(
        &p / 0,
        c_in,
        c_out,
        ksize,
        ConvConfig {
            stride,
            padding,
            groups,
            bias: false,
            ..Default::default()
        },
    );
    let norm = batch_norm2d(
        &p / 1,
        c_out,
        BatchNormConfig {
            eps: 0.001,
            ..Default::default()
        },
    );
    let s: Vec<&str> = p.components().collect();
    let s = s.join(".");

    nn::func_t(move |xs, train| {
        let ys = xs.apply(&conv);

        let ys = ys.apply_t(&norm, train);

        let ys = if activation { ys.silu() } else { ys };
        // if s == "features.1.0.block.0" {
        //     println!("{ys}");
        // }

        ys
    })
}

fn squeeze_excitation(p: nn::Path, c_in: i64, c_squeeze: i64) -> impl nn::ModuleT {
    let fc1 = conv2d(&p / "fc1", c_in, c_squeeze, 1, Default::default());
    let fc2 = conv2d(&p / "fc2", c_squeeze, c_in, 1, Default::default());
    nn::func_t(move |xs, _| {
        let scale = xs
            .adaptive_avg_pool2d([1, 1])
            .apply(&fc1)
            .silu()
            .apply(&fc2)
            .sigmoid();
        xs * scale
    })
}

enum StochasticDepthKind {
    Row,
    Batch,
}

fn stochastic_depth(prob: f64, kind: StochasticDepthKind) -> impl nn::ModuleT {
    nn::func_t(move |xs, train| {
        if !train || prob == 0.0 {
            return xs.shallow_clone();
        }
        let survival_rate = 1.0 - prob;
        let size = match kind {
            StochasticDepthKind::Row => {
                let mut size = vec![xs.size()[0]];
                size.extend(std::iter::repeat(1).take(xs.dim() - 1));
                size
            }
            StochasticDepthKind::Batch => std::iter::repeat(1).take(xs.dim()).collect(),
        };
        let noise = Tensor::rand(size, (tch::Kind::Float, xs.device()));
        let noise = noise.ge(survival_rate).to_kind(tch::Kind::Float);
        if survival_rate > 0.0 {
            xs * noise / survival_rate
        } else {
            xs * noise
        }
    })
}

fn mbconv(p: nn::Path, cfg: MBConvConfig, stochastic_depth_prob: f64) -> impl nn::ModuleT {
    let use_res_connect = cfg.stride == 1 && cfg.c_in == cfg.c_out;
    let c_expanded = MBConvConfig::adjust_channels(cfg.c_in, cfg.expand_ratio, None);
    let mut layers = nn::seq_t();
    let q = &p / "block";

    if c_expanded != cfg.c_in {
        let expand = conv_norm_activation(&q / layers.len(), cfg.c_in, c_expanded, 1, 1, 1, true);
        layers = layers.add(expand);
    }

    let depthwise = conv_norm_activation(
        &q / layers.len(),
        c_expanded,
        c_expanded,
        cfg.ksize,
        cfg.stride,
        c_expanded,
        true,
    );
    layers = layers.add(depthwise);

    let c_squeeze = i64::max(1, cfg.c_in / 4);
    let se_layer = squeeze_excitation(&q / layers.len(), c_expanded, c_squeeze);
    layers = layers.add(se_layer);

    let project = conv_norm_activation(&q / layers.len(), c_expanded, cfg.c_out, 1, 1, 1, false);
    layers = layers.add(project);

    let stochastic_depth = stochastic_depth(stochastic_depth_prob, StochasticDepthKind::Row);

    nn::func_t(move |xs, train| {
        let ys = xs.apply_t(&layers, train);
        if use_res_connect {
            xs + ys.apply_t(&stochastic_depth, train)
        } else {
            ys
        }
    })
}

fn fused_mbconv(p: nn::Path, cfg: MBConvConfig, stochastic_depth_prob: f64) -> impl nn::ModuleT {
    let use_res_connect = (cfg.stride == 1) && (cfg.c_in == cfg.c_out);
    let c_expanded = MBConvConfig::adjust_channels(cfg.c_in, cfg.expand_ratio, None);
    let mut layers = nn::seq_t();
    let q = &p / "block";

    if c_expanded != cfg.c_in {
        let fused_expand = conv_norm_activation(
            &q / layers.len(),
            cfg.c_in,
            c_expanded,
            cfg.ksize,
            cfg.stride,
            1,
            true,
        );
        layers = layers.add(fused_expand);
        let project =
            conv_norm_activation(&q / layers.len(), c_expanded, cfg.c_out, 1, 1, 1, false);
        layers = layers.add(project);
    } else {
        let layer = conv_norm_activation(
            &q / layers.len(),
            cfg.c_in,
            cfg.c_out,
            cfg.ksize,
            cfg.stride,
            1,
            true,
        );
        layers = layers.add(layer);
    }

    let stochastic_depth = stochastic_depth(stochastic_depth_prob, StochasticDepthKind::Row);

    nn::func_t(move |xs, train| {
        let ys = xs.apply_t(&layers, train);
        if use_res_connect {
            xs + ys.apply_t(&stochastic_depth, train)
        } else {
            ys
        }
    })
}

fn efficientnet(
    p: &nn::Path,
    cfgs: &[MBConvConfig],
    dropout: f64,
    stochastic_depth_prob: f64,
    num_classes: i64,
    last_channel: Option<i64>,
) -> impl nn::ModuleT {
    let mut layers = nn::seq_t();
    let q = p / "features";

    let firstconv_c_out = cfgs[0].c_in;
    layers = layers.add(conv_norm_activation(
        &q / 0,
        3,
        firstconv_c_out,
        3,
        2,
        1,
        true,
    ));

    let total_stage_blocks = cfgs.iter().map(|cfg| cfg.num_layers).sum::<i64>();
    let mut stage_block_idx = 0;

    for (i, cfg) in cfgs.iter().enumerate() {
        let mut stage = nn::seq_t();
        for j in 0..cfg.num_layers {
            let mut block_cfg = *cfg;
            if j != 0 {
                block_cfg.c_in = block_cfg.c_out;
                block_cfg.stride = 1;
            }

            let sd_prob =
                stochastic_depth_prob * (stage_block_idx as f64) / total_stage_blocks as f64;
            stage = match block_cfg.block_type {
                BlockType::MBConv => stage.add(mbconv(&q / (i + 1) / j, block_cfg, sd_prob)),
                BlockType::FusedMBConv => {
                    stage.add(fused_mbconv(&q / (i + 1) / j, block_cfg, sd_prob))
                }
            };
            stage_block_idx += 1;
        }
        layers = layers.add(stage);
    }

    let lastconv_c_in = cfgs.last().unwrap().c_out;
    let lastconv_c_out = last_channel.unwrap_or(4 * lastconv_c_in);
    let lastconv = conv_norm_activation(
        &q / layers.len(),
        lastconv_c_in,
        lastconv_c_out,
        1,
        1,
        1,
        true,
    );
    layers = layers.add(lastconv);

    let fc = linear(
        p / "classifier" / 1,
        lastconv_c_out,
        num_classes,
        Default::default(),
    );

    nn::seq_t()
        .add(layers)
        .add_fn(|xs| xs.adaptive_avg_pool2d([1, 1]).flat_view())
        .add_fn_t(move |xs, train| xs.dropout(dropout, train))
        .add(fc)
}

#[derive(Debug, Clone, Copy)]
enum Arch {
    B0,
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
    V2S,
    V2M,
    V2L,
}

fn efficientnet_conf(
    arch: Arch,
    width_mult: Option<f64>,
    depth_mult: Option<f64>,
) -> (Vec<MBConvConfig>, Option<i64>) {
    let width_mult = width_mult.unwrap_or(1.0);
    let depth_mult = depth_mult.unwrap_or(1.0);

    let cfgs = match arch {
        Arch::V2S => vec![
            MBConvConfig::fused_mbconv(1.0, 3, 1, 24, 24, 2),
            MBConvConfig::fused_mbconv(4.0, 3, 2, 24, 48, 4),
            MBConvConfig::fused_mbconv(4.0, 3, 2, 48, 64, 4),
            MBConvConfig::mbconv(4.0, 3, 2, 64, 128, 6, width_mult, depth_mult),
            MBConvConfig::mbconv(6.0, 3, 1, 128, 160, 9, width_mult, depth_mult),
            MBConvConfig::mbconv(6.0, 3, 2, 160, 256, 15, width_mult, depth_mult),
        ],
        Arch::V2M => vec![
            MBConvConfig::fused_mbconv(1.0, 3, 1, 24, 24, 3),
            MBConvConfig::fused_mbconv(4.0, 3, 2, 24, 48, 5),
            MBConvConfig::fused_mbconv(4.0, 3, 2, 48, 80, 5),
            MBConvConfig::mbconv(4.0, 3, 2, 80, 160, 7, width_mult, depth_mult),
            MBConvConfig::mbconv(6.0, 3, 1, 160, 176, 14, width_mult, depth_mult),
            MBConvConfig::mbconv(6.0, 3, 2, 176, 304, 18, width_mult, depth_mult),
            MBConvConfig::mbconv(6.0, 3, 1, 304, 512, 5, width_mult, depth_mult),
        ],
        Arch::V2L => vec![
            MBConvConfig::fused_mbconv(1.0, 3, 1, 32, 32, 4),
            MBConvConfig::fused_mbconv(4.0, 3, 2, 32, 64, 7),
            MBConvConfig::fused_mbconv(4.0, 3, 2, 64, 96, 7),
            MBConvConfig::mbconv(4.0, 3, 2, 96, 192, 10, width_mult, depth_mult),
            MBConvConfig::mbconv(6.0, 3, 1, 192, 224, 19, width_mult, depth_mult),
            MBConvConfig::mbconv(6.0, 3, 2, 224, 384, 25, width_mult, depth_mult),
            MBConvConfig::mbconv(6.0, 3, 1, 384, 640, 7, width_mult, depth_mult),
        ],
        _ => vec![
            MBConvConfig::mbconv(1.0, 3, 1, 32, 16, 1, width_mult, depth_mult),
            MBConvConfig::mbconv(6.0, 3, 2, 16, 24, 2, width_mult, depth_mult),
            MBConvConfig::mbconv(6.0, 5, 2, 24, 40, 2, width_mult, depth_mult),
            MBConvConfig::mbconv(6.0, 3, 2, 40, 80, 3, width_mult, depth_mult),
            MBConvConfig::mbconv(6.0, 5, 1, 80, 112, 3, width_mult, depth_mult),
            MBConvConfig::mbconv(6.0, 5, 2, 112, 192, 4, width_mult, depth_mult),
            MBConvConfig::mbconv(6.0, 3, 1, 192, 320, 1, width_mult, depth_mult),
        ],
    };
    let last_channel = match arch {
        Arch::V2S | Arch::V2M | Arch::V2L => Some(1280),
        _ => None,
    };
    (cfgs, last_channel)
}

pub fn efficientnet_b0(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let (cfgs, last_channel) = efficientnet_conf(Arch::B0, Some(1.0), Some(1.0));
    efficientnet(p, &cfgs, 0.2, 0.2, num_classes, last_channel)
}

pub fn efficientnet_b1(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let (cfgs, last_channel) = efficientnet_conf(Arch::B1, Some(1.0), Some(1.1));
    efficientnet(p, &cfgs, 0.2, 0.2, num_classes, last_channel)
}

pub fn efficientnet_b2(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let (cfgs, last_channel) = efficientnet_conf(Arch::B2, Some(1.1), Some(1.2));
    efficientnet(p, &cfgs, 0.3, 0.2, num_classes, last_channel)
}

pub fn efficientnet_b3(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let (cfgs, last_channel) = efficientnet_conf(Arch::B3, Some(1.2), Some(1.4));
    efficientnet(p, &cfgs, 0.3, 0.2, num_classes, last_channel)
}

pub fn efficientnet_b4(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let (cfgs, last_channel) = efficientnet_conf(Arch::B4, Some(1.4), Some(1.8));
    efficientnet(p, &cfgs, 0.4, 0.2, num_classes, last_channel)
}

pub fn efficientnet_b5(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let (cfgs, last_channel) = efficientnet_conf(Arch::B5, Some(1.6), Some(2.2));
    efficientnet(p, &cfgs, 0.4, 0.2, num_classes, last_channel)
}

pub fn efficientnet_b6(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let (cfgs, last_channel) = efficientnet_conf(Arch::B6, Some(1.8), Some(2.6));
    efficientnet(p, &cfgs, 0.5, 0.2, num_classes, last_channel)
}

pub fn efficientnet_b7(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let (cfgs, last_channel) = efficientnet_conf(Arch::B7, Some(2.0), Some(3.1));
    efficientnet(p, &cfgs, 0.5, 0.2, num_classes, last_channel)
}

pub fn efficientnet_v2s(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let (cfgs, last_channel) = efficientnet_conf(Arch::V2S, None, None);
    efficientnet(p, &cfgs, 0.2, 0.2, num_classes, last_channel)
}

pub fn efficientnet_v2m(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let (cfgs, last_channel) = efficientnet_conf(Arch::V2M, None, None);
    efficientnet(p, &cfgs, 0.3, 0.2, num_classes, last_channel)
}

pub fn efficientnet_v2l(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let (cfgs, last_channel) = efficientnet_conf(Arch::V2L, None, None);
    efficientnet(p, &cfgs, 0.4, 0.2, num_classes, last_channel)
}
