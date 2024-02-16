use tch::{
    nn::{self, LayerNormConfig, LinearConfig},
    Tensor,
};

fn layernorm2d(
    p: nn::Path,
    normalized_shape: Vec<i64>,
    config: nn::LayerNormConfig,
) -> impl nn::ModuleT {
    nn::seq_t()
        .add_fn(|xs| xs.permute([0, 2, 3, 1]))
        .add(nn::layer_norm(p, normalized_shape, config))
        .add_fn(|xs| xs.permute([0, 3, 1, 2]))
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
fn cn_block(
    p: nn::Path,
    dim: i64,
    layer_scale: f64,
    stochastic_depth_prob: f64,
) -> impl nn::ModuleT {
    let layer_scale = p.var_copy(
        "layer_scale",
        &(layer_scale * &Tensor::ones([dim, 1, 1], (tch::Kind::Float, p.device()))),
    );
    let stoch_depth = stochastic_depth(stochastic_depth_prob, StochasticDepthKind::Row);
    let p = p / "block";
    let block = nn::seq_t()
        .add(nn::conv2d(
            &p / 0,
            dim,
            dim,
            7,
            nn::ConvConfig {
                padding: 3,
                groups: dim,
                ..Default::default()
            },
        ))
        .add_fn(|xs| xs.permute([0, 2, 3, 1]))
        .add(nn::layer_norm(
            &p / 2,
            vec![dim],
            LayerNormConfig::default(),
        ))
        .add(nn::linear(&p / 3, dim, 4 * dim, LinearConfig::default()))
        .add_fn(|xs| xs.gelu("none"))
        .add(nn::linear(&p / 5, 4 * dim, dim, LinearConfig::default()))
        .add_fn(|xs| xs.permute([0, 3, 1, 2]));
    nn::func_t(move |xs, train| {
        let result = &layer_scale * xs.apply_t(&block, train);
        let result = &result.apply_t(&stoch_depth, train);
        result + xs
    })
}

fn conv_norm(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    ksize: i64,
    stride: i64,
    padding: i64,
) -> impl nn::ModuleT {
    nn::seq_t()
        .add(nn::conv2d(
            &p / 0,
            c_in,
            c_out,
            ksize,
            nn::ConvConfig {
                stride,
                padding,
                ..Default::default()
            },
        ))
        .add(layernorm2d(&p / 1, vec![c_out], LayerNormConfig::default()))
}

fn convnext(
    p: &nn::Path,
    block_setting: &Vec<(i64, i64, i64)>,
    stochastic_depth_prob: f64,
    layer_scale: f64,
    num_classes: i64,
) -> impl nn::ModuleT {
    let mut layers = nn::seq_t();
    let firsconv_output_channels = block_setting[0].0;
    let feats = p / "features";

    layers = layers.add(conv_norm(&feats / 0, 3, firsconv_output_channels, 4, 4, 0));
    let total_stage_blocks = block_setting.iter().map(|x| x.2).sum::<i64>();
    let mut stage_block_idx = 0;

    for (c_in, c_out, num_layers) in block_setting {
        let mut stage = nn::seq_t();
        for i in 0..*num_layers {
            let sd_prob = stochastic_depth_prob * (stage_block_idx as f64)
                / (total_stage_blocks as f64 - 1.0);
            stage = stage.add(cn_block(
                &feats / layers.len() / i,
                *c_in,
                layer_scale,
                sd_prob,
            ));
            stage_block_idx += 1;
        }
        layers = layers.add(stage);
        if *c_out != -1 {
            let q = &feats / layers.len();
            layers = layers.add(
                nn::seq_t()
                    .add(layernorm2d(&q / 0, vec![*c_in], LayerNormConfig::default()))
                    .add(nn::conv2d(
                        &q / 1,
                        *c_in,
                        *c_out,
                        2,
                        nn::ConvConfig {
                            stride: 2,
                            ..Default::default()
                        },
                    )),
            );
        }
    }
    let lastblock = block_setting.last().unwrap();
    let lastconv_output_channels = if lastblock.1 != -1 {
        lastblock.1
    } else {
        lastblock.0
    };
    let classifier = nn::seq_t()
        .add(layernorm2d(
            p / "classifier" / 0,
            vec![lastconv_output_channels],
            LayerNormConfig::default(),
        ))
        .add_fn(|xs| xs.flatten(1, -1))
        .add(nn::linear(
            p / "classifier" / 2,
            lastconv_output_channels,
            num_classes,
            Default::default(),
        ));
    nn::seq_t()
        .add(layers)
        .add_fn(|xs| xs.adaptive_avg_pool2d([1, 1]))
        .add(classifier)
}

pub fn convnext_tiny(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    convnext(
        p,
        &vec![(96, 192, 3), (192, 384, 3), (384, 768, 9), (768, -1, 3)],
        0.1,
        1e-6,
        num_classes,
    )
}

pub fn convnext_small(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    convnext(
        p,
        &vec![(96, 192, 3), (192, 384, 3), (384, 768, 27), (768, -1, 3)],
        0.4,
        1e-6,
        num_classes,
    )
}

pub fn convnext_base(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    convnext(
        p,
        &vec![(128, 256, 3), (256, 512, 3), (512, 1024, 27), (1024, -1, 3)],
        0.5,
        1e-6,
        num_classes,
    )
}

pub fn convnext_large(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    convnext(
        p,
        &vec![(192, 384, 3), (384, 768, 3), (768, 1536, 27), (1536, -1, 3)],
        0.5,
        1e-6,
        num_classes,
    )
}
