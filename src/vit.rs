use tch::{
    nn::{self, conv2d, layer_norm, linear, ConvConfig, LayerNormConfig},
    IndexOp, Kind, Tensor,
};

fn mlp_block(p: nn::Path, in_dim: i64, mlp_dim: i64, dropout: f64) -> impl nn::ModuleT {
    nn::seq_t()
        .add(linear(&p / 0, in_dim, mlp_dim, Default::default()))
        .add_fn_t(move |xs, train| xs.gelu("none").dropout(dropout, train))
        .add(linear(&p / 3, mlp_dim, in_dim, Default::default()))
        .add_fn_t(move |xs, train| xs.dropout(dropout, train))
}

fn attention(p: nn::Path, dim: i64, num_heads: i64) -> impl nn::ModuleT {
    let in_proj_weight = p.var(
        "in_proj_weight",
        &[dim * 3, dim],
        nn::Init::Randn {
            mean: 0.0,
            stdev: 0.02,
        },
    );
    let in_proj_bias = p.var("in_proj_bias", &[dim * 3], nn::Init::Const(0.));
    let out_proj = linear(&p / "out_proj", dim, dim, Default::default());
    let scale = 1. / ((dim / num_heads) as f64).sqrt();
    nn::func_t(move |xs, _| {
        let (b, n, c) = xs.size3().unwrap();
        let qkv = (xs.matmul(&in_proj_weight.transpose(0, 1)) + &in_proj_bias)
            .reshape([b, n, 3, num_heads, c / num_heads])
            .permute([2, 0, 3, 1, 4]);
        let (q, k, v) = (qkv.get(0) * scale, qkv.get(1), qkv.get(2));
        let attn = q.matmul(&k.transpose(-2, -1)).softmax(-1, Kind::Float);
        attn.matmul(&v)
            .transpose(1, 2)
            .reshape([b, n, c])
            .apply(&out_proj)
    })
}

fn encoder_block(
    p: nn::Path,
    num_heads: i64,
    hidden_dim: i64,
    mlp_dim: i64,
    dropout: f64,
) -> impl nn::ModuleT {
    let ln1 = layer_norm(
        &p / "ln_1",
        vec![hidden_dim],
        LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        },
    );
    let self_attention = attention(&p / "self_attention", hidden_dim, num_heads);
    let ln2 = layer_norm(
        &p / "ln_2",
        vec![hidden_dim],
        LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        },
    );
    let mlp = mlp_block(p / "mlp", hidden_dim, mlp_dim, dropout);

    nn::func_t(move |xs, train| {
        let x = xs
            .apply(&ln1)
            .apply_t(&self_attention, train)
            .dropout(dropout, train)
            + xs;
        let y = x.apply(&ln2).apply_t(&mlp, train);
        x + y
    })
}

fn encoder(
    p: nn::Path,
    seq_length: i64,
    num_layers: i64,
    num_heads: i64,
    hidden_dim: i64,
    mlp_dim: i64,
    dropout: f64,
) -> impl nn::ModuleT {
    let pos_embedding = p.var(
        "pos_embedding",
        &[1, seq_length, hidden_dim],
        nn::Init::Randn {
            mean: 0.0,
            stdev: 0.02,
        },
    );
    let mut layers = nn::seq_t();
    for i in 0..num_layers {
        layers = layers.add(encoder_block(
            &p / "layers" / format!("encoder_layer_{i}"),
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
        ));
    }
    let ln = layer_norm(
        &p / "ln",
        vec![hidden_dim],
        LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        },
    );

    nn::func_t(move |xs, train| {
        let xs = xs + &pos_embedding;
        xs.dropout(dropout, train)
            .apply_t(&layers, train)
            .apply(&ln)
    })
}

fn vision_transformer(
    p: &nn::Path,
    image_size: i64,
    patch_size: i64,
    num_layers: i64,
    num_heads: i64,
    hidden_dim: i64,
    mlp_dim: i64,
    dropout: f64,
    num_classes: i64,
) -> impl nn::ModuleT {
    let conv_proj = conv2d(
        p / "conv_proj",
        3,
        hidden_dim,
        patch_size,
        ConvConfig {
            stride: patch_size,
            ..Default::default()
        },
    );
    let seq_length = (image_size / patch_size) * (image_size / patch_size) + 1;
    let class_token = p.var("class_token", &[1, 1, hidden_dim], nn::Init::Const(0.));

    let encoder = encoder(
        p / "encoder",
        seq_length,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout,
    );

    let head = linear(
        p / "heads" / "head",
        hidden_dim,
        num_classes,
        Default::default(),
    );

    nn::func_t(move |xs, train| {
        let (b, _, h, w) = xs.size4().unwrap();
        let (n_h, n_w) = (h / patch_size, w / patch_size);
        let xs = xs
            .apply(&conv_proj)
            .reshape([b, hidden_dim, n_h * n_w])
            .permute([0, 2, 1]);
        let batch_class_token = class_token.expand([b, -1, -1], true);
        Tensor::cat(&[batch_class_token, xs], 1)
            .apply_t(&encoder, train)
            .i((.., 0))
            .apply(&head)
    })
}

pub fn vit_b_16(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    vision_transformer(p, 224, 16, 12, 12, 768, 3072, 0.0, num_classes)
}

pub fn vit_b_32(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    vision_transformer(p, 224, 32, 12, 12, 768, 3072, 0.0, num_classes)
}

pub fn vit_l_16(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    vision_transformer(p, 224, 16, 24, 16, 1024, 4096, 0.0, num_classes)
}

pub fn vit_l_32(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    vision_transformer(p, 224, 32, 24, 16, 1024, 4096, 0.0, num_classes)
}

pub fn vit_h_14(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    vision_transformer(p, 224, 14, 32, 16, 1280, 5120, 0.0, num_classes)
}

