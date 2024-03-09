/* Ported from torch vision library
 * RegNet model architecture from
 * `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.
 */

use std::collections::HashSet;

use tch::nn::{self, batch_norm2d, conv2d, BatchNormConfig, ConvConfig};

fn make_divisible(v: f64, divisor: i64, min_value: Option<i64>) -> i64 {
    let min_value = min_value.unwrap_or(divisor);
    let new_v = i64::max(
        min_value,
        (v + divisor as f64 / 2.0) as i64 / divisor * divisor,
    );
    if (new_v as f64) < (0.9 * v) {
        new_v + divisor
    } else {
        new_v
    }
}

#[derive(Debug, Clone)]
enum Activation {
    Relu,
    None,
}

fn conv_norm_activation(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    ksize: i64,
    stride: i64,
    groups: i64,
    dilation: i64,
    activation: Activation,
) -> impl nn::ModuleT {
    let padding = (ksize - 1) / 2 * dilation;
    nn::seq_t()
        .add(conv2d(
            &p / 0,
            c_in,
            c_out,
            ksize,
            ConvConfig {
                stride,
                groups,
                padding,
                bias: false,
                dilation,
                ws_init: nn::Init::Kaiming {
                    dist: nn::init::NormalOrUniform::Normal,
                    fan: nn::init::FanInOut::FanOut,
                    non_linearity: nn::init::NonLinearity::ReLU,
                },
                ..Default::default()
            },
        ))
        .add(batch_norm2d(
            &p / 1,
            c_out,
            BatchNormConfig {
                ws_init: nn::Init::Const(1.0),
                ..Default::default()
            },
        ))
        .add_fn(move |xs| match activation {
            Activation::Relu => xs.relu(),
            Activation::None => xs.shallow_clone(),
        })
}

fn simple_stem_in(
    p: nn::Path,
    width_in: i64,
    width_out: i64,
    activation: Activation,
) -> impl nn::ModuleT {
    conv_norm_activation(p, width_in, width_out, 3, 2, 1, 1, activation)
}

fn squeeze_excitation(p: nn::Path, c_in: i64, c_squeeze: i64) -> impl nn::ModuleT {
    let scale = nn::seq_t()
        .add_fn(|xs| xs.adaptive_avg_pool2d([1, 1]))
        .add(conv2d(
            &p / "fc1",
            c_in,
            c_squeeze,
            1,
            ConvConfig {
                ws_init: nn::Init::Kaiming {
                    dist: nn::init::NormalOrUniform::Normal,
                    fan: nn::init::FanInOut::FanOut,
                    non_linearity: nn::init::NonLinearity::ReLU,
                },
                ..Default::default()
            },
        ))
        .add_fn(|xs| xs.relu())
        .add(conv2d(
            &p / "fc2",
            c_squeeze,
            c_in,
            1,
            ConvConfig {
                ws_init: nn::Init::Kaiming {
                    dist: nn::init::NormalOrUniform::Normal,
                    fan: nn::init::FanInOut::FanOut,
                    non_linearity: nn::init::NonLinearity::ReLU,
                },
                ..Default::default()
            },
        ))
        .add_fn(|xs| xs.sigmoid());
    nn::func_t(move |xs, train| xs * xs.apply_t(&scale, train))
}

fn bottleneck_transform(
    p: nn::Path,
    width_in: i64,
    width_out: i64,
    stride: i64,
    group_width: i64,
    bottleneck_multiplier: f64,
    se_ratio: Option<f64>,
) -> impl nn::ModuleT {
    let w_b = (width_out as f64 * bottleneck_multiplier).round() as i64;
    let g = w_b / group_width;

    let mut layers = nn::seq_t()
        .add(conv_norm_activation(
            &p / "a",
            width_in,
            w_b,
            1,
            1,
            1,
            1,
            Activation::Relu,
        ))
        .add(conv_norm_activation(
            &p / "b",
            w_b,
            w_b,
            3,
            stride,
            g,
            1,
            Activation::Relu,
        ));
    if se_ratio.is_some() {
        let width_se_out = (width_in as f64 * se_ratio.unwrap()).round() as i64;
        layers = layers.add(squeeze_excitation(&p / "se", w_b, width_se_out));
    }
    layers = layers.add(conv_norm_activation(
        &p / "c",
        w_b,
        width_out,
        1,
        1,
        1,
        1,
        Activation::None,
    ));
    layers
}

fn any_stage(
    p: nn::Path,
    width_in: i64,
    width_out: i64,
    stride: i64,
    depth: i64,
    group_width: i64,
    bottleneck_multiplier: f64,
    se_ratio: Option<f64>,
    stage_index: i64,
) -> impl nn::ModuleT {
    let mut layers = nn::seq_t();
    for i in 0..depth {
        let width_in = if i == 0 { width_in } else { width_out };
        let stride = if i == 0 { stride } else { 1 };
        layers = layers.add(res_bottleneck_block(
            &p / format!("block{stage_index}-{i}"),
            width_in,
            width_out,
            stride,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        ));
    }
    layers
}

#[derive(Debug, Clone)]
struct BlockParams {
    depths: Vec<i64>,
    widths: Vec<i64>,
    group_widths: Vec<i64>,
    bottleneck_multipliers: Vec<f64>,
    strides: Vec<i64>,
    se_ratio: Option<f64>,
}

impl BlockParams {
    fn from_init_params(
        depth: i64,
        w_0: i64,
        w_a: f64,
        w_m: f64,
        group_width: i64,
        bottleneck_multiplier: f64,
        se_ratio: Option<f64>,
    ) -> Self {
        let quant = 8.;
        let stride = 2;
        let widths_cont: Vec<f64> = (0..depth).map(|i| i as f64 * w_a + w_0 as f64).collect();
        let block_capacity: Vec<f64> = widths_cont
            .iter()
            .map(|&w| ((w / w_0 as f64).ln() / w_m.ln()).round())
            .collect();
        let block_widths: Vec<i64> = block_capacity
            .iter()
            .map(|&c| (w_0 as f64 * w_m.powi(c as i32) / quant - 1e-6).round() * quant) // Slight
            // hack to prevent rounding error, maybe should fix
            .map(|w| w as i64)
            .collect();

        let num_stages: HashSet<_> = block_widths.iter().cloned().collect::<_>();
        let num_stages = num_stages.len();

        let mut b0 = block_widths.clone();
        b0.push(0);
        let mut b1 = vec![0];
        b1.extend(block_widths.clone());

        let split_helper: Vec<(i64, i64, i64, i64)> = (0..block_widths.len() + 1)
            .map(|i| (b0[i], b1[i], b0[i], b1[i]))
            .collect();

        let split_helper = split_helper
            .iter()
            .map(|(w, wp, r, rp)| (w != wp) || (r != rp))
            .collect::<Vec<_>>();

        let stage_widths: Vec<_> = block_widths
            .iter()
            .zip(split_helper.iter().take(split_helper.len() - 1))
            .filter_map(|(&w, &t)| if t { Some(w) } else { None })
            .collect();

        let stage_depths: Vec<i64> = split_helper
            .iter()
            .enumerate()
            .filter_map(|(d, &t)| if t { Some(d as i64) } else { None })
            .collect::<Vec<_>>()
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        let strides = vec![stride; num_stages];
        let bottleneck_multipliers = vec![bottleneck_multiplier; num_stages];
        let group_widths = vec![group_width; num_stages];

        let (stage_widths, group_widths) = Self::adjust_widths_group_compatibility(
            stage_widths,
            &bottleneck_multipliers,
            group_widths,
        );

        Self {
            depths: stage_depths,
            widths: stage_widths,
            group_widths,
            bottleneck_multipliers,
            strides,
            se_ratio,
        }
    }

    fn adjust_widths_group_compatibility(
        stage_widths: Vec<i64>,
        bottleneck_ratios: &[f64],
        group_widths: Vec<i64>,
    ) -> (Vec<i64>, Vec<i64>) {
        let widths = stage_widths
            .iter()
            .zip(bottleneck_ratios.iter())
            .map(|(&w, &b)| (w as f64 * b).round() as i64)
            .collect::<Vec<_>>();
        let group_widths_min: Vec<i64> = group_widths
            .iter()
            .zip(widths.iter())
            .map(|(&g, &w)| g.min(w))
            .collect();

        let ws_bot: Vec<i64> = widths
            .iter()
            .zip(group_widths_min.iter())
            .map(|(&w, &g)| make_divisible(w as f64, g, None))
            .collect();
        let stage_widths = ws_bot
            .iter()
            .zip(bottleneck_ratios.iter())
            .map(|(&w, &b)| (w as f64 / b).round() as i64)
            .collect();
        (stage_widths, group_widths_min)
    }

    fn get_expanded_params(&self) -> Vec<(i64, i64, i64, i64, f64)> {
        self.widths
            .iter()
            .zip(&self.strides)
            .zip(&self.depths)
            .zip(&self.group_widths)
            .zip(&self.bottleneck_multipliers)
            .map(
                |((((&width, &stride), &depth), &group_width), &bottleneck_multiplier)| {
                    (width, stride, depth, group_width, bottleneck_multiplier)
                },
            )
            .collect()
    }
}

fn res_bottleneck_block(
    p: nn::Path,
    width_in: i64,
    width_out: i64,
    stride: i64,
    group_width: i64,
    bottleneck_multiplier: f64,
    se_ratio: Option<f64>,
) -> impl nn::ModuleT {
    let projection = if width_in != width_out || stride != 1 {
        Some(conv_norm_activation(
            &p / "proj",
            width_in,
            width_out,
            1,
            stride,
            1,
            1,
            Activation::None,
        ))
    } else {
        None
    };
    let f = bottleneck_transform(
        &p / "f",
        width_in,
        width_out,
        stride,
        group_width,
        bottleneck_multiplier,
        se_ratio,
    );

    nn::func_t(move |xs, train| {
        let ys = xs.apply_t(&f, train);
        let ys = match &projection {
            Some(proj) => xs.apply_t(proj, train) + ys,
            None => xs + ys,
        };
        ys.relu()
    })
}

fn regnet(
    p: &nn::Path,
    block_params: BlockParams,
    num_classes: i64,
    stem_width: i64,
) -> impl nn::ModuleT {
    let stem = simple_stem_in(p / "stem", 3, stem_width, Activation::Relu);
    let mut current_width = stem_width;
    let q = p / "trunk_output";
    let mut blocks = nn::seq_t();
    for (i, (width, stride, depth, group_width, bottleneck_multiplier)) in
        block_params.get_expanded_params().iter().enumerate()
    {
        blocks = blocks.add(any_stage(
            &q / format!("block{}", i + 1),
            current_width,
            *width,
            *stride,
            *depth,
            *group_width,
            *bottleneck_multiplier,
            block_params.se_ratio,
            (i + 1) as i64,
        ));
        current_width = *width;
    }
    let fc = nn::linear(
        p / "fc",
        current_width,
        num_classes,
        nn::LinearConfig {
            ws_init: nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
            ..Default::default()
        },
    );

    nn::func_t(move |xs, train| {
        xs.apply_t(&stem, train)
            .apply_t(&blocks, train)
            .adaptive_avg_pool2d([1, 1])
            .flat_view()
            .apply(&fc)
    })
}

pub fn regnet_y_400mf(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let block_params = BlockParams::from_init_params(16, 48, 27.89, 2.09, 8, 1.0, Some(0.25));
    regnet(p, block_params, num_classes, 32)
}

pub fn regnet_y_800mf(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let block_params = BlockParams::from_init_params(14, 56, 38.84, 2.4, 16, 1.0, Some(0.25));
    regnet(p, block_params, num_classes, 32)
}

pub fn regnet_y_1_6gf(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let block_params = BlockParams::from_init_params(27, 48, 20.71, 2.65, 24, 1.0, Some(0.25));
    regnet(p, block_params, num_classes, 32)
}

pub fn regnet_y_3_2gf(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let block_params = BlockParams::from_init_params(21, 80, 42.63, 2.66, 24, 1.0, Some(0.25));
    regnet(p, block_params, num_classes, 32)
}

pub fn regnet_y_8gf(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let block_params = BlockParams::from_init_params(17, 192, 76.82, 2.19, 56, 1.0, Some(0.25));
    regnet(p, block_params, num_classes, 32)
}

pub fn regnet_y_16gf(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let block_params = BlockParams::from_init_params(18, 200, 106.23, 2.48, 112, 1.0, Some(0.25));
    regnet(p, block_params, num_classes, 32)
}

pub fn regnet_y_32gf(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let block_params = BlockParams::from_init_params(20, 232, 115.89, 2.53, 232, 1.0, Some(0.25));
    regnet(p, block_params, num_classes, 32)
}

pub fn regnet_y_128gf(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let block_params = BlockParams::from_init_params(27, 456, 160.83, 2.52, 264, 1.0, Some(0.25));
    regnet(p, block_params, num_classes, 32)
}

pub fn regnet_x_400mf(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let block_params = BlockParams::from_init_params(22, 24, 24.48, 2.54, 16, 1.0, None);
    regnet(p, block_params, num_classes, 32)
}

pub fn regnet_x_800mf(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let block_params = BlockParams::from_init_params(16, 56, 35.73, 2.28, 16, 1.0, None);
    regnet(p, block_params, num_classes, 32)
}

pub fn regnet_x_1_6gf(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let block_params = BlockParams::from_init_params(18, 80, 34.01, 2.25, 24, 1.0, None);
    regnet(p, block_params, num_classes, 32)
}

pub fn regnet_x_3_2gf(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let block_params = BlockParams::from_init_params(25, 88, 26.31, 2.25, 48, 1.0, None);
    regnet(p, block_params, num_classes, 32)
}

pub fn regnet_x_8gf(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let block_params = BlockParams::from_init_params(23, 80, 49.56, 2.88, 120, 1.0, None);
    regnet(p, block_params, num_classes, 32)
}

pub fn regnet_x_16gf(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let block_params = BlockParams::from_init_params(22, 216, 55.59, 2.1, 128, 1.0, None);
    regnet(p, block_params, num_classes, 32)
}

pub fn regnet_x_32gf(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let block_params = BlockParams::from_init_params(23, 320, 69.86, 2.0, 168, 1.0, None);
    regnet(p, block_params, num_classes, 32)
}
