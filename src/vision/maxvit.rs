use std::ops::{AddAssign, MulAssign};

use tch::{
    nn::{
        self, batch_norm2d, conv2d, layer_norm, linear, BatchNormConfig, ConvConfig, LinearConfig,
    },
    Device, IndexOp, Kind, Tensor,
};

enum NormLayer {
    BatchNorm,
    LayerNorm,
}

struct ConvNormActivationConfig {
    ksize: i64,
    stride: i64,
    padding: Option<i64>,
    groups: i64,
    activation: bool,
    bias: Option<bool>,
    norm_layer: Option<NormLayer>,
    dilation: i64,
}

impl Default for ConvNormActivationConfig {
    fn default() -> Self {
        Self {
            ksize: 3,
            stride: 1,
            padding: None,
            groups: 1,
            activation: true,
            bias: None,
            norm_layer: Some(NormLayer::BatchNorm),
            dilation: 1,
        }
    }
}

fn conv_norm_activation(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    cfg: ConvNormActivationConfig,
) -> impl nn::ModuleT {
    let padding = cfg.padding.unwrap_or((cfg.ksize - 1) / 2 * cfg.dilation);
    let bias = cfg.bias.unwrap_or(cfg.norm_layer.is_none());
    let mut layers = nn::seq_t().add(conv2d(
        &p / 0,
        c_in,
        c_out,
        cfg.ksize,
        ConvConfig {
            stride: cfg.stride,
            padding,
            groups: cfg.groups,
            bias,
            dilation: cfg.dilation,
            ..Default::default()
        },
    ));
    if let Some(norm_layer) = cfg.norm_layer {
        match norm_layer {
            NormLayer::BatchNorm => {
                layers = layers.add(batch_norm2d(
                    &p / 1,
                    c_out,
                    BatchNormConfig {
                        eps: 1e-3,
                        momentum: 0.99,
                        ..Default::default()
                    },
                ));
            }
            NormLayer::LayerNorm => {
                layers = layers.add(nn::layer_norm(&p / 1, vec![c_out], Default::default()));
            }
        }
    }
    if cfg.activation {
        layers = layers.add_fn(|xs| xs.gelu("none"));
    }
    layers
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

fn mbconv(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    expansion_ratio: f64,
    squeeze_ratio: f64,
    stride: i64,
    p_stochastic_dropout: f64,
) -> impl nn::ModuleT {
    let should_proj = c_in != c_out || stride != 1;
    let mut proj = nn::seq();
    if should_proj {
        if stride == 2 {
            proj = proj
                .add_fn(move |xs| xs.avg_pool2d(3, stride, 1, false, true, None))
                .add(conv2d(&p / "proj" / 1, c_in, c_out, 1, Default::default()));
        } else {
            proj = proj.add(conv2d(&p / "proj" / 0, c_in, c_out, 1, Default::default()));
        }
    }

    let c_mid = (c_out as f64 * expansion_ratio) as i64;
    let c_squeeze = (c_out as f64 * squeeze_ratio) as i64;

    let pre_norm = batch_norm2d(
        &p / "layers" / "pre_norm",
        c_in,
        BatchNormConfig {
            eps: 1e-3,
            momentum: 0.99,
            ..Default::default()
        },
    );

    let conv_a = conv_norm_activation(
        &p / "layers" / "conv_a",
        c_in,
        c_mid,
        ConvNormActivationConfig {
            ksize: 1,
            ..Default::default()
        },
    );

    let conv_b = conv_norm_activation(
        &p / "layers" / "conv_b",
        c_mid,
        c_mid,
        ConvNormActivationConfig {
            ksize: 3,
            stride,
            padding: Some(1),
            groups: c_mid,
            ..Default::default()
        },
    );

    let se_layer = squeeze_excitation(&p / "layers" / "squeeze_excitation", c_mid, c_squeeze);
    let conv_c = conv2d(
        &p / "layers" / "conv_c",
        c_mid,
        c_out,
        1,
        Default::default(),
    );
    let stoch_depth = stochastic_depth(p_stochastic_dropout, StochasticDepthKind::Row);

    nn::func_t(move |xs, train| {
        let residual = xs.apply_t(&proj, train);
        residual
            + xs.apply_t(&pre_norm, train)
                .apply_t(&conv_a, train)
                .apply_t(&conv_b, train)
                .apply_t(&se_layer, train)
                .apply_t(&conv_c, train)
                .apply_t(&stoch_depth, train)
    })
}

fn get_relative_position_index(height: i64, width: i64) -> Tensor {
    let coords = Tensor::stack(
        &Tensor::meshgrid(&[
            &Tensor::arange(height, (Kind::Int, Device::Cpu)),
            &Tensor::arange(width, (Kind::Int, Device::Cpu)),
        ]),
        0,
    )
    .flatten(1, -1);
    let mut relative_coords = coords.unsqueeze(-1) - coords.unsqueeze(1);
    relative_coords = relative_coords.permute([1, 2, 0]).contiguous();
    relative_coords.i((.., .., 0)).add_assign(height - 1);
    relative_coords.i((.., .., 1)).add_assign(width - 1);
    relative_coords.i((.., .., 0)).mul_assign(2 * width - 2);
    relative_coords.sum_dim_intlist(-1, false, Kind::Float)
}

fn get_relative_positional_bias(rpi: &Tensor, rpb_table: &Tensor, max_seq_len: i64) -> Tensor {
    let bias_index = rpi.view([-1]);
    rpb_table
        .i((&bias_index,))
        .view([max_seq_len, max_seq_len, -1])
        .permute([2, 0, 1])
        .contiguous()
        .unsqueeze(0)
}

fn relative_positional_multi_head_attention(
    p: nn::Path,
    feat_dim: i64,
    head_dim: i64,
    max_seq_len: i64,
) -> impl nn::ModuleT {
    let n_heads = feat_dim / head_dim;
    let size = (max_seq_len as f64).sqrt() as i64;
    let to_qkv = linear(
        &p / "to_qkv",
        feat_dim,
        n_heads * head_dim * 3,
        Default::default(),
    );

    let scale_factor = (feat_dim as f64).powf(-0.5);

    let merge = linear(
        &p / "merge",
        n_heads * head_dim,
        feat_dim,
        Default::default(),
    );

    let relative_position_bias_table = p.var(
        "relative_position_bias_table",
        &[(2 * size - 1) * (2 * size - 1), n_heads],
        nn::Init::Randn {
            mean: 0.0,
            stdev: 0.02,
        },
    );

    let rpi = get_relative_position_index(size, size);
    let mut relative_position_index = p.ones_no_train("relative_position_index", &rpi.size());
    relative_position_index.copy_(&rpi);

    nn::func_t(move |xs, _| {
        let (b, g, p, d) = xs.size4().unwrap();

        let qkv = xs.apply(&to_qkv).chunk(3, -1);
        let q = qkv[0]
            .reshape([b, g, p, n_heads, head_dim])
            .permute([0, 1, 3, 2, 4]);
        let k = qkv[1]
            .reshape([b, g, p, n_heads, head_dim])
            .permute([0, 1, 3, 2, 4])
            * scale_factor;
        let v = qkv[2]
            .reshape([b, g, p, n_heads, head_dim])
            .permute([0, 1, 3, 2, 4]);

        let dot_prod = Tensor::einsum("b g h i d, b g h j d -> b g h i j", &[q, k], None::<i64>);
        let pos_bias = get_relative_positional_bias(
            &relative_position_index,
            &relative_position_bias_table,
            max_seq_len,
        );
        let dot_prod = (dot_prod + pos_bias).softmax(-1, Kind::Float);
        Tensor::einsum(
            "b g h i j, b g h j d -> b g h i d",
            &[dot_prod, v],
            None::<i64>,
        )
        .permute([0, 1, 3, 2, 4])
        .reshape([b, g, p, d])
        .apply(&merge)
    })
}

fn window_partition(x: &Tensor, p: i64) -> Tensor {
    let (b, c, h, w) = x.size4().unwrap();
    x.reshape([b, c, h / p, p, w / p, p])
        .permute([0, 2, 4, 3, 5, 1])
        .reshape([b, (h / p) * (w / p), p * p, c])
}

fn window_departition(x: &Tensor, p: i64, h_partitions: i64, w_partitions: i64) -> Tensor {
    let (b, _g, _pp, c) = x.size4().unwrap();
    x.reshape([b, h_partitions, w_partitions, p, p, c])
        .permute([0, 5, 1, 3, 2, 4])
        .reshape([b, c, h_partitions * p, w_partitions * p])
}

enum PartitionType {
    Grid,
    Window,
}

fn partition_attention_layer(
    p: nn::Path,
    c_in: i64,
    head_dim: i64,
    partition_size: i64,
    partition_type: PartitionType,
    grid_size: (i64, i64),
    mlp_ratio: i64,
    attention_dropout: f64,
    mlp_dropout: f64,
    p_stochastic_dropout: f64,
) -> impl nn::ModuleT {
    let n_partitions = grid_size.0 / partition_size;
    let (ps, _gs) = match partition_type {
        PartitionType::Grid => (n_partitions, partition_size),
        PartitionType::Window => (partition_size, n_partitions),
    };
    let attn_layer = nn::seq_t()
        .add(layer_norm(
            &p / "attn_layer" / 0,
            vec![c_in],
            Default::default(),
        ))
        .add(relative_positional_multi_head_attention(
            &p / "attn_layer" / 1,
            c_in,
            head_dim,
            partition_size * partition_size,
        ))
        .add_fn_t(move |xs, train| xs.dropout(attention_dropout, train));
    let mlp_layer = nn::seq_t()
        .add(layer_norm(
            &p / "mlp_layer" / 0,
            vec![c_in],
            Default::default(),
        ))
        .add(linear(
            &p / "mlp_layer" / 1,
            c_in,
            c_in * mlp_ratio,
            Default::default(),
        ))
        .add_fn(|xs| xs.gelu("none"))
        .add(linear(
            &p / "mlp_layer" / 3,
            c_in * mlp_ratio,
            c_in,
            Default::default(),
        ))
        .add_fn_t(move |xs, train| xs.dropout(mlp_dropout, train));

    let stoch_drop = stochastic_depth(p_stochastic_dropout, StochasticDepthKind::Row);

    nn::func_t(move |xs, train| {
        let (gh, gw) = (grid_size.0 / ps, grid_size.1 / ps);
        let mut ys = window_partition(xs, ps);
        if let PartitionType::Grid = partition_type {
            ys = ys.swapaxes(-2, -3);
        }
        ys = &ys + ys.apply_t(&attn_layer, train).apply_t(&stoch_drop, train);
        ys = &ys + ys.apply_t(&mlp_layer, train).apply_t(&stoch_drop, train);
        if let PartitionType::Grid = partition_type {
            ys = ys.swapaxes(-2, -3);
        }

        window_departition(&ys, ps, gh, gw)
    })
}

fn max_vit_layer(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    squeeze_ratio: f64,
    expansion_ratio: f64,
    stride: i64,
    head_dim: i64,
    mlp_ratio: i64,
    mlp_dropout: f64,
    attention_dropout: f64,
    p_stochastic_dropout: f64,
    partition_size: i64,
    grid_size: (i64, i64),
) -> impl nn::ModuleT {
    nn::seq_t()
        .add(mbconv(
            &p / "layers" / "MBconv",
            c_in,
            c_out,
            expansion_ratio,
            squeeze_ratio,
            stride,
            p_stochastic_dropout,
        ))
        .add(partition_attention_layer(
            &p / "layers" / "window_attention",
            c_out,
            head_dim,
            partition_size,
            PartitionType::Window,
            grid_size,
            mlp_ratio,
            attention_dropout,
            mlp_dropout,
            p_stochastic_dropout,
        ))
        .add(partition_attention_layer(
            &p / "layers" / "grid_attention",
            c_out,
            head_dim,
            partition_size,
            PartitionType::Grid,
            grid_size,
            mlp_ratio,
            attention_dropout,
            mlp_dropout,
            p_stochastic_dropout,
        ))
}

fn get_conv_output_shape(
    input_size: (i64, i64),
    ksize: i64,
    stride: i64,
    padding: i64,
) -> (i64, i64) {
    (
        (input_size.0 + 2 * padding - ksize) / stride + 1,
        (input_size.1 + 2 * padding - ksize) / stride + 1,
    )
}

fn max_vit_block(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    squeeze_ratio: f64,
    expansion_ratio: f64,
    head_dim: i64,
    mlp_ratio: i64,
    mlp_dropout: f64,
    attention_dropout: f64,
    partition_size: i64,
    input_grid_size: (i64, i64),
    p_stochastic: &[f64],
) -> impl nn::ModuleT {
    let grid_size = get_conv_output_shape(input_grid_size, 3, 2, 1);
    let mut layers = nn::seq_t();
    for (idx, prob) in p_stochastic.iter().enumerate() {
        let stride = if idx == 0 { 2 } else { 1 };
        layers = layers.add(max_vit_layer(
            &p / "layers" / idx,
            if idx == 0 { c_in } else { c_out },
            c_out,
            squeeze_ratio,
            expansion_ratio,
            stride,
            head_dim,
            mlp_ratio,
            mlp_dropout,
            attention_dropout,
            *prob,
            partition_size,
            grid_size,
        ));
    }
    layers
}

fn make_block_input_shapes(input_size: (i64, i64), n_blocks: i64) -> Vec<(i64, i64)> {
    let mut shapes = vec![];
    let mut size = get_conv_output_shape(input_size, 3, 2, 1);
    for _ in 0..n_blocks {
        size = get_conv_output_shape(size, 3, 2, 1);
        shapes.push(size);
    }
    shapes
}

fn max_vit(
    p: &nn::Path,
    input_size: (i64, i64),
    stem_channels: i64,
    partition_size: i64,
    block_channels: &[i64],
    block_layers: &[i64],
    head_dim: i64,
    stochastic_depth_prob: f64,
    squeeze_ratio: f64,
    expansion_ratio: f64,
    mlp_ratio: i64,
    mlp_dropout: f64,
    attention_dropout: f64,
    num_classes: i64,
) -> impl nn::ModuleT {
    let c_in = 3;

    let stem = nn::seq_t()
        .add(conv_norm_activation(
            p / "stem" / 0,
            c_in,
            stem_channels,
            ConvNormActivationConfig {
                ksize: 3,
                stride: 2,
                activation: true,
                bias: Some(false),
                norm_layer: Some(NormLayer::BatchNorm),
                ..Default::default()
            },
        ))
        .add(conv_norm_activation(
            p / "stem" / 1,
            stem_channels,
            stem_channels,
            ConvNormActivationConfig {
                ksize: 3,
                stride: 1,
                activation: false,
                bias: Some(true),
                norm_layer: None,
                ..Default::default()
            },
        ));

    let mut input_size = get_conv_output_shape(input_size, 3, 2, 1);
    let mut in_channels = vec![stem_channels];
    in_channels.extend(&block_channels[..block_channels.len() - 1]);
    let out_channels = block_channels;

    // p_stochastic = np.linspace(0, stochastic_depth_prob, sum(block_layers)).tolist()
    let p_stochastic = vec![0.0; block_layers.iter().sum::<i64>() as usize]; // TODO calculate this
                                                                             // properly

    let mut p_idx: usize = 0;
    let mut blocks = nn::seq_t();

    for i in 0..block_layers.len() {
        blocks = blocks.add(max_vit_block(
            p / "blocks" / i,
            in_channels[i],
            out_channels[i],
            squeeze_ratio,
            expansion_ratio,
            head_dim,
            mlp_ratio,
            mlp_dropout,
            attention_dropout,
            partition_size,
            input_size,
            &p_stochastic[p_idx..(p_idx + block_layers[i] as usize)],
        ));
        input_size = get_conv_output_shape(input_size, 3, 2, 1);
        p_idx += block_layers[i] as usize;
    }
    let dim = *block_channels.last().unwrap();
    let classifier = nn::seq_t()
        .add_fn(|xs| xs.adaptive_avg_pool2d([1, 1]).flat_view())
        .add(layer_norm(
            p / "classifier" / 2,
            vec![dim],
            Default::default(),
        ))
        .add(linear(p / "classifier" / 3, dim, dim, Default::default()))
        .add_fn(|xs| xs.tanh())
        .add(linear(
            p / "classifier" / 5,
            dim,
            num_classes,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        ));

    nn::func_t(move |xs, train| {
        xs.apply_t(&stem, train)
            .apply_t(&blocks, train)
            .apply_t(&classifier, train)
    })
}

pub fn maxvit_t(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    max_vit(
        p,
        (224, 224),
        64,
        7,
        &[64, 128, 256, 512],
        &[2, 2, 5, 2],
        32,
        0.2,
        0.25,
        4.0,
        4,
        0.0,
        0.0,
        num_classes,
    )
}
