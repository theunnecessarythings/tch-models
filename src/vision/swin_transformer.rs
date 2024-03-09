/* Ported from torchvision library
* Swin Transformer model architecture from
* `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/abs/2103.14030>`_.
*/

use std::ops::{AddAssign, DivAssign, MulAssign};

use tch::{
    nn::{self, conv2d, layer_norm, linear, ConvConfig, LayerNormConfig, LinearConfig},
    no_grad_guard, Device, IndexOp, Kind, Tensor,
};

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

fn mlp_block(p: nn::Path, in_dim: i64, mlp_dim: i64, dropout: f64) -> impl nn::ModuleT {
    nn::seq_t()
        .add(linear(&p / 0, in_dim, mlp_dim, Default::default()))
        .add_fn_t(move |xs, train| xs.gelu("none").dropout(dropout, train))
        .add(linear(&p / 3, mlp_dim, in_dim, Default::default()))
        .add_fn_t(move |xs, train| xs.dropout(dropout, train))
}

fn patch_merging_pad(x: &Tensor) -> Tensor {
    let size = x.dim();
    let (h, w) = (x.size()[size - 3], x.size()[size - 2]);
    let x = x.pad([0, 0, 0, w % 2, 0, h % 2], "constant", 0.0);

    let h_indices_0 = Tensor::arange_start_step(0, x.size()[1], 2, (Kind::Int, x.device()));
    let w_indices_0 = Tensor::arange_start_step(0, x.size()[2], 2, (Kind::Int, x.device()));
    let h_indices_1 = Tensor::arange_start_step(1, x.size()[1], 2, (Kind::Int, x.device()));
    let w_indices_1 = Tensor::arange_start_step(1, x.size()[2], 2, (Kind::Int, x.device()));

    let x0 = x
        .index_select(1, &h_indices_0)
        .index_select(2, &w_indices_0);
    let x1 = x
        .index_select(1, &h_indices_1)
        .index_select(2, &w_indices_0);
    let x2 = x
        .index_select(1, &h_indices_0)
        .index_select(2, &w_indices_1);
    let x3 = x
        .index_select(1, &h_indices_1)
        .index_select(2, &w_indices_1);

    Tensor::cat(&[x0, x1, x2, x3], -1)
}

fn patch_merging(p: nn::Path, dim: i64) -> impl nn::ModuleT {
    let reduction = linear(
        &p / "reduction",
        4 * dim,
        2 * dim,
        LinearConfig {
            bias: false,
            ..Default::default()
        },
    );
    let norm = layer_norm(&p / "norm", vec![4 * dim], Default::default());
    nn::func_t(move |xs, _| patch_merging_pad(xs).apply(&norm).apply(&reduction))
}

fn patch_merging_v2(p: nn::Path, dim: i64) -> impl nn::ModuleT {
    let reduction = linear(
        &p / "reduction",
        4 * dim,
        2 * dim,
        LinearConfig {
            bias: false,
            ..Default::default()
        },
    );
    let norm = layer_norm(&p / "norm", vec![2 * dim], Default::default());
    nn::func_t(move |xs, _| patch_merging_pad(xs).apply(&reduction).apply(&norm))
}

fn normalize(x: &Tensor, p: f64, dim: i64, eps: f64) -> Tensor {
    let norm = x
        .norm_scalaropt_dim(p, dim, true)
        .clamp_min(eps)
        .expand_as(x);
    x / norm
}

fn shifted_window_attention(
    xs: &Tensor,
    qkv_weight: &Tensor,
    proj_weight: &Tensor,
    relative_position_bias: &Tensor,
    window_size: [i64; 2],
    num_heads: i64,
    shift_size: Vec<i64>,
    attention_dropout: f64,
    dropout: f64,
    qkv_bias: Option<&Tensor>,
    proj_bias: Option<&Tensor>,
    logit_scale: Option<&Tensor>,
    training: bool,
) -> Tensor {
    let (b, h, w, c) = xs.size4().unwrap();
    let pad_r = (window_size[1] - w % window_size[1]) % window_size[1];
    let pad_b = (window_size[0] - h % window_size[0]) % window_size[0];
    let mut xs = xs.pad([0, 0, 0, pad_r, 0, pad_b], "constant", 0.0);
    let (_, pad_h, pad_w, _) = xs.size4().unwrap();

    let mut shift_size = shift_size.clone();
    if window_size[0] >= pad_h {
        shift_size[0] = 0;
    }
    if window_size[1] >= pad_w {
        shift_size[1] = 0;
    }

    let shift_size_sum = shift_size.iter().sum::<i64>();
    if shift_size_sum > 0 {
        xs = xs.roll([-shift_size[0], -shift_size[1]], [1, 2]);
    }

    let num_windows = (pad_h / window_size[0]) * (pad_w / window_size[1]);
    xs = xs.view([
        b,
        pad_h / window_size[0],
        window_size[0],
        pad_w / window_size[1],
        window_size[1],
        c,
    ]);
    xs = xs.permute([0, 1, 3, 2, 4, 5]).reshape([
        b * num_windows,
        window_size[0] * window_size[1],
        c,
    ]);
    let mut qkv;
    if logit_scale.is_some() && qkv_bias.is_some() {
        let _no_grad = no_grad_guard();
        let qkv_bias = qkv_bias.unwrap();
        let length = (qkv_bias.numel() / 3) as i64;
        let _ = qkv_bias.i(length..(2 * length)).zero_();
        qkv = xs.linear(qkv_weight, Some(qkv_bias));
    } else {
        qkv = xs.linear(qkv_weight, qkv_bias);
    }
    qkv = qkv
        .reshape([xs.size()[0], xs.size()[1], 3, num_heads, c / num_heads])
        .permute([2, 0, 3, 1, 4]);
    let (mut q, k, v) = (qkv.i(0), qkv.i(1), qkv.i(2));

    let mut attn;
    if logit_scale.is_some() {
        attn =
            normalize(&q, 2.0, -1, 1e-12).matmul(&normalize(&k, 2.0, -1, 1e-12).transpose(-2, -1));
        attn *= logit_scale.unwrap().clamp_max(100.0_f64.ln()).exp();
    } else {
        q *= ((c / num_heads) as f64).powf(-0.5);
        attn = q.matmul(&k.transpose(-2, -1));
    }
    attn += relative_position_bias;

    if shift_size_sum > 0 {
        let mut attn_mask = xs.new_zeros([pad_h, pad_w], (Kind::Float, xs.device()));
        let h_slices = [
            (0, pad_w - window_size[0]),
            (pad_h - window_size[0], pad_w - shift_size[0]),
            (pad_h - shift_size[0], pad_h),
        ];
        let w_slices = [
            (0, pad_w - window_size[1]),
            (pad_h - window_size[1], pad_w - shift_size[1]),
            (pad_h - shift_size[1], pad_w),
        ];
        let mut count = 0;
        for h in h_slices {
            for w in w_slices {
                let _ = attn_mask.i((h.0..h.1, w.0..w.1)).fill_(count);
                count += 1;
            }
        }
        attn_mask = attn_mask
            .view([
                pad_h / window_size[0],
                window_size[0],
                pad_w / window_size[1],
                window_size[1],
            ])
            .permute([0, 2, 1, 3])
            .reshape([num_windows, window_size[0] * window_size[1]]);
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2);
        attn_mask = attn_mask
            .masked_fill(&attn_mask.ne(0.0), -100.0)
            .masked_fill(&attn_mask.eq(0.0), 0.0);
        attn = attn.view([
            xs.size()[0] / num_windows,
            num_windows,
            num_heads,
            xs.size()[1],
            xs.size()[1],
        ]);
        attn += attn_mask.unsqueeze(1).unsqueeze(0);
        attn = attn.view([-1, num_heads, xs.size()[1], xs.size()[1]]);
    }

    attn = attn
        .softmax(-1, Kind::Float)
        .dropout(attention_dropout, training);

    xs = attn
        .matmul(&v)
        .transpose(1, 2)
        .reshape([xs.size()[0], xs.size()[1], c])
        .linear(proj_weight, proj_bias)
        .dropout(dropout, training)
        .view([
            b,
            pad_h / window_size[0],
            pad_w / window_size[1],
            window_size[0],
            window_size[1],
            c,
        ])
        .permute([0, 1, 3, 2, 4, 5])
        .reshape([b, pad_h, pad_w, c]);

    if shift_size_sum > 0 {
        xs = xs.roll([shift_size[0], shift_size[1]], [1, 2]);
    }

    xs.i((.., ..h, ..w, ..)).contiguous()
}

fn define_relative_position_index(window_size: [i64; 2]) -> Tensor {
    let coords_h = Tensor::arange(window_size[0], (Kind::Int64, tch::Device::Cpu));
    let coords_w = Tensor::arange(window_size[1], (Kind::Int64, tch::Device::Cpu));
    let coords =
        Tensor::stack(&Tensor::meshgrid_indexing(&[&coords_h, &coords_w], "ij"), 0).flatten(1, -1);

    let mut relative_coords = coords.unsqueeze(-1) - coords.unsqueeze(1);
    relative_coords = relative_coords.permute([1, 2, 0]).contiguous();
    relative_coords
        .i((.., .., 0))
        .add_assign(window_size[0] - 1);
    relative_coords
        .i((.., .., 1))
        .add_assign(window_size[1] - 1);
    relative_coords
        .i((.., .., 0))
        .mul_assign(2 * window_size[1] - 2);
    relative_coords
        .sum_dim_intlist(-1, false, Kind::Float)
        .flatten(0, -1)
}

fn get_relative_position_bias(
    relative_position_bias_table: &Tensor,
    relative_position_index: &Tensor,
    window_size: [i64; 2],
) -> Tensor {
    let n = window_size[0] * window_size[1];
    relative_position_bias_table
        .i(relative_position_index)
        .view([n, n, -1])
        .permute([2, 0, 1])
        .contiguous()
        .unsqueeze(0)
}

fn shifted_window_attention_block(
    p: nn::Path,
    dim: i64,
    window_size: [i64; 2],
    shift_size: Vec<i64>,
    num_heads: i64,
    qkv_bias: bool,
    proj_bias: bool,
    attention_dropout: f64,
    dropout: f64,
) -> impl nn::ModuleT {
    let qkv = linear(
        &p / "qkv",
        dim,
        dim * 3,
        LinearConfig {
            bias: qkv_bias,
            ..Default::default()
        },
    );
    let proj = linear(
        &p / "proj",
        dim,
        dim,
        LinearConfig {
            bias: proj_bias,
            ..Default::default()
        },
    );

    let relative_position_bias_table = p.var(
        "relative_position_bias_table",
        &[
            (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
            num_heads,
        ],
        nn::Init::Randn {
            mean: 0.0,
            stdev: 0.02,
        },
    );

    let rpi = define_relative_position_index(window_size);
    let mut relative_position_index = p.ones_no_train("relative_position_index", &rpi.size());
    relative_position_index.copy_(&rpi);

    nn::func_t(move |xs, train| {
        let relative_position_bias = get_relative_position_bias(
            &relative_position_bias_table,
            &relative_position_index,
            window_size,
        );
        shifted_window_attention(
            xs,
            qkv.ws.as_ref(),
            proj.ws.as_ref(),
            &relative_position_bias,
            window_size,
            num_heads,
            shift_size.clone(),
            attention_dropout,
            dropout,
            qkv.bs.as_ref(),
            proj.bs.as_ref(),
            None,
            train,
        )
    })
}

fn define_relative_position_bias_table_v2(window_size: [i64; 2]) -> Tensor {
    let relative_coords_h = Tensor::arange_start(
        -(window_size[0] - 1),
        window_size[0],
        (Kind::Float, Device::Cpu),
    );
    let relative_coords_w = Tensor::arange_start(
        -(window_size[1] - 1),
        window_size[1],
        (Kind::Float, Device::Cpu),
    );
    let mut relative_coords_table = Tensor::stack(
        &Tensor::meshgrid(&[relative_coords_h, relative_coords_w]),
        0,
    )
    .permute([1, 2, 0])
    .contiguous()
    .unsqueeze(0);

    relative_coords_table
        .i((.., .., .., 0))
        .div_assign(window_size[0] - 1);
    relative_coords_table
        .i((.., .., .., 1))
        .div_assign(window_size[1] - 1);

    relative_coords_table.mul_assign(8);

    relative_coords_table.sign() * (relative_coords_table.abs() + 1.0).log2() / 3.0
}

fn shifted_window_attention_block_v2(
    p: nn::Path,
    dim: i64,
    window_size: [i64; 2],
    shift_size: Vec<i64>,
    num_heads: i64,
    qkv_bias: bool,
    proj_bias: bool,
    attention_dropout: f64,
    dropout: f64,
) -> impl nn::ModuleT {
    let mut qkv = linear(
        &p / "qkv",
        dim,
        dim * 3,
        LinearConfig {
            bias: qkv_bias,
            ..Default::default()
        },
    );
    let proj = linear(
        &p / "proj",
        dim,
        dim,
        LinearConfig {
            bias: proj_bias,
            ..Default::default()
        },
    );

    let rct = define_relative_position_bias_table_v2(window_size);
    let mut relative_coords_table = p.ones_no_train("relative_coords_table", &rct.size());
    relative_coords_table.copy_(&rct);

    let rpi = define_relative_position_index(window_size);
    let mut relative_position_index = p.ones_no_train("relative_position_index", &rpi.size());
    relative_position_index.copy_(&rpi);

    let logit_scale = p.var(
        "logit_scale",
        &[num_heads, 1, 1],
        nn::Init::Const(10.0_f64.ln()),
    );

    let cpb_mlp = nn::seq()
        .add(linear(&p / "cpb_mlp" / 0, 2, 512, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(linear(
            &p / "cpb_mlp" / 2,
            512,
            num_heads,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        ));

    if qkv_bias {
        let _no_grad = no_grad_guard();
        let length = (qkv.bs.as_ref().unwrap().numel() / 3) as i64;
        let _ = qkv.bs.as_mut().unwrap().i(length..(2 * length)).zero_();
    }

    nn::func_t(move |xs, train| {
        let relative_position_bias = get_relative_position_bias(
            &relative_coords_table.apply(&cpb_mlp).view([-1, num_heads]),
            &relative_position_index,
            window_size,
        )
        .sigmoid()
            * 16;
        shifted_window_attention(
            xs,
            qkv.ws.as_ref(),
            proj.ws.as_ref(),
            &relative_position_bias,
            window_size,
            num_heads,
            shift_size.clone(),
            attention_dropout,
            dropout,
            qkv.bs.as_ref(),
            proj.bs.as_ref(),
            Some(&logit_scale),
            train,
        )
    })
}

fn swin_transformer_block(
    p: nn::Path,
    dim: i64,
    num_heads: i64,
    window_size: [i64; 2],
    shift_size: Vec<i64>,
    mlp_ratio: f64,
    dropout: f64,
    attention_dropout: f64,
    stochastic_depth_prob: f64,
) -> impl nn::ModuleT {
    let norm1 = layer_norm(
        &p / "norm1",
        vec![dim],
        LayerNormConfig {
            eps: 1e-5,
            ..Default::default()
        },
    );
    let attn = shifted_window_attention_block(
        &p / "attn",
        dim,
        window_size,
        shift_size,
        num_heads,
        true,
        true,
        attention_dropout,
        dropout,
    );
    let stoch_depth = stochastic_depth(stochastic_depth_prob, StochasticDepthKind::Row);
    let norm2 = layer_norm(&p / "norm2", vec![dim], Default::default());
    let mlp = mlp_block(&p / "mlp", dim, (dim as f64 * mlp_ratio) as i64, dropout);
    nn::func_t(move |xs, train| {
        let residual = xs.shallow_clone();
        let xs = xs
            .apply(&norm1)
            .apply_t(&attn, train)
            .apply_t(&stoch_depth, train);
        let xs = xs + &residual;
        let residual = xs.shallow_clone();
        let xs = xs
            .apply(&norm2)
            .apply_t(&mlp, train)
            .apply_t(&stoch_depth, train);
        xs + residual
    })
}

fn swin_transformer_block_v2(
    p: nn::Path,
    dim: i64,
    num_heads: i64,
    window_size: [i64; 2],
    shift_size: Vec<i64>,
    mlp_ratio: f64,
    dropout: f64,
    attention_dropout: f64,
    stochastic_depth_prob: f64,
) -> impl nn::ModuleT {
    let norm1 = layer_norm(&p / "norm1", vec![dim], Default::default());
    let attn = shifted_window_attention_block_v2(
        &p / "attn",
        dim,
        window_size,
        shift_size,
        num_heads,
        true,
        true,
        attention_dropout,
        dropout,
    );
    let stoch_depth = stochastic_depth(stochastic_depth_prob, StochasticDepthKind::Row);
    let norm2 = layer_norm(&p / "norm2", vec![dim], Default::default());
    let mlp = mlp_block(&p / "mlp", dim, (dim as f64 * mlp_ratio) as i64, dropout);
    nn::func_t(move |xs, train| {
        let residual = xs.shallow_clone();
        let xs = xs
            .apply_t(&attn, train)
            .apply(&norm1)
            .apply_t(&stoch_depth, train);
        let xs = xs + &residual;
        let residual = xs.shallow_clone();
        let xs = xs
            .apply_t(&mlp, train)
            .apply(&norm2)
            .apply_t(&stoch_depth, train);
        xs + residual
    })
}
fn swin_transformer(
    p: &nn::Path,
    patch_size: i64,
    embed_dim: i64,
    depths: &[i64],
    num_heads: &[i64],
    window_size: [i64; 2],
    mlp_ratio: f64,
    dropout: f64,
    attention_dropout: f64,
    stochastic_depth_prob: f64,
    num_classes: i64,
    v2: bool,
) -> impl nn::ModuleT {
    let mut layers = nn::seq_t().add(
        nn::seq_t()
            .add(conv2d(
                p / "features" / 0 / 0,
                3,
                embed_dim,
                patch_size,
                ConvConfig {
                    stride: patch_size,
                    ..Default::default()
                },
            ))
            .add_fn(|xs| xs.permute([0, 2, 3, 1]))
            .add(layer_norm(
                p / "features" / 0 / 2,
                vec![embed_dim],
                LayerNormConfig {
                    eps: 1e-5,
                    ..Default::default()
                },
            )),
    );

    let total_stage_blocks = depths.iter().sum::<i64>();
    let mut stage_block_idx = 0;
    for i_stage in 0..depths.len() {
        let mut stage = nn::seq_t();
        let dim = embed_dim * 2_i64.pow(i_stage as u32);
        for i_layer in 0..depths[i_stage] {
            let sd_prob =
                stochastic_depth_prob * (stage_block_idx as f64) / (total_stage_blocks - 1) as f64;
            let ss = window_size
                .iter()
                .map(|&x| if i_layer % 2 == 0 { 0 } else { x / 2 })
                .collect();
            if v2 {
                let block = swin_transformer_block_v2(
                    p / "features" / layers.len() / i_layer,
                    dim,
                    num_heads[i_stage],
                    window_size,
                    ss,
                    mlp_ratio,
                    dropout,
                    attention_dropout,
                    sd_prob,
                );
                stage = stage.add(block);
            } else {
                let block = swin_transformer_block(
                    p / "features" / layers.len() / i_layer,
                    dim,
                    num_heads[i_stage],
                    window_size,
                    ss,
                    mlp_ratio,
                    dropout,
                    attention_dropout,
                    sd_prob,
                );
                stage = stage.add(block);
            }
            stage_block_idx += 1;
        }
        layers = layers.add(stage);
        if i_stage < depths.len() - 1 {
            if v2 {
                let layer = patch_merging_v2(p / "features" / layers.len(), dim);
                layers = layers.add(layer);
            } else {
                let layer = patch_merging(p / "features" / layers.len(), dim);
                layers = layers.add(layer);
            }
        }
    }
    let num_features = embed_dim * 2_i64.pow(depths.len() as u32 - 1);
    let norm = layer_norm(
        p / "norm",
        vec![num_features],
        LayerNormConfig {
            eps: 1e-5,
            ..Default::default()
        },
    );
    let head = linear(p / "head", num_features, num_classes, Default::default());

    nn::func_t(move |xs, train| {
        xs.apply_t(&layers, train)
            .apply(&norm)
            .permute([0, 3, 1, 2])
            .adaptive_avg_pool2d([1, 1])
            .flat_view()
            .apply(&head)
    })
}

pub fn swin_t(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    swin_transformer(
        p,
        4,
        96,
        &[2, 2, 6, 2],
        &[3, 6, 12, 24],
        [7, 7],
        4.0,
        0.0,
        0.0,
        0.2,
        num_classes,
        false,
    )
}

pub fn swin_s(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    swin_transformer(
        p,
        4,
        96,
        &[2, 2, 18, 2],
        &[3, 6, 12, 24],
        [7, 7],
        4.0,
        0.0,
        0.0,
        0.3,
        num_classes,
        false,
    )
}

pub fn swin_b(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    swin_transformer(
        p,
        4,
        128,
        &[2, 2, 18, 2],
        &[4, 8, 16, 32],
        [7, 7],
        4.0,
        0.0,
        0.0,
        0.5,
        num_classes,
        false,
    )
}

pub fn swin_v2_t(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    swin_transformer(
        p,
        4,
        96,
        &[2, 2, 6, 2],
        &[3, 6, 12, 24],
        [8, 8],
        4.0,
        0.0,
        0.0,
        0.2,
        num_classes,
        true,
    )
}

pub fn swin_v2_s(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    swin_transformer(
        p,
        4,
        96,
        &[2, 2, 18, 2],
        &[3, 6, 12, 24],
        [8, 8],
        4.0,
        0.0,
        0.0,
        0.3,
        num_classes,
        true,
    )
}

pub fn swin_v2_b(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    swin_transformer(
        p,
        4,
        128,
        &[2, 2, 18, 2],
        &[4, 8, 16, 32],
        [8, 8],
        4.0,
        0.0,
        0.0,
        0.5,
        num_classes,
        true,
    )
}
