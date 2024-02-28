use tch::{nn, IndexOp, Tensor};

fn feed_forward(p: nn::Path, dim: i64, hidden_dim: i64, dropout: f64) -> impl nn::ModuleT {
    let p = &p / "net";
    nn::seq_t()
        .add(nn::layer_norm(&p / 0, vec![dim], Default::default()))
        .add(nn::linear(&p / 1, dim, hidden_dim, Default::default()))
        .add_fn_t(move |xs, train| xs.gelu("none").dropout(dropout, train))
        .add(nn::linear(&p / 4, hidden_dim, dim, Default::default()))
        .add_fn_t(move |xs, train| xs.dropout(dropout, train))
}

fn attention(p: nn::Path, dim: i64, heads: i64, head_dim: i64, dropout: f64) -> impl nn::ModuleT {
    let inner_dim = head_dim * heads;
    let project_out = !(heads == 1 && head_dim == dim);
    let scale = 1. / (head_dim as f64).sqrt();
    let norm = nn::layer_norm(&p / "norm", vec![dim], Default::default());
    let to_qkv = nn::linear(
        &p / "to_qkv",
        dim,
        inner_dim * 3,
        nn::LinearConfig {
            bias: false,
            ..Default::default()
        },
    );
    let mut to_out = nn::seq_t();
    if project_out {
        to_out = to_out
            .add(nn::linear(
                &p / "to_out" / 0,
                inner_dim,
                dim,
                Default::default(),
            ))
            .add_fn_t(move |xs, train| xs.dropout(dropout, train));
    }

    nn::func_t(move |xs, train| {
        let qkv = xs
            .apply(&norm)
            .apply(&to_qkv)
            .chunk(3, -1)
            .iter()
            .map(|x| {
                x.reshape([-1, x.size()[1], heads, head_dim])
                    .transpose(1, 2)
            })
            .collect::<Vec<_>>();
        let (q, k, v) = (&qkv[0], &qkv[1], &qkv[2]);
        let dots = q.matmul(&k.transpose(-1, -2)) * scale;
        let out = dots
            .softmax(-1, tch::Kind::Float)
            .dropout(dropout, train)
            .matmul(v);
        out.transpose(1, 2)
            .reshape([-1, out.size()[2], inner_dim])
            .apply_t(&to_out, train)
    })
}

fn transformer(
    p: nn::Path,
    dim: i64,
    depth: i64,
    heads: i64,
    head_dim: i64,
    mlp_dim: i64,
    dropout: f64,
) -> impl nn::ModuleT {
    let mut layers = vec![];
    let p = p / "layers";
    for i in 0..depth {
        layers.push((
            attention(&p / i / 0, dim, heads, head_dim, dropout),
            feed_forward(&p / i / 1, dim, mlp_dim, dropout),
        ));
    }
    nn::func_t(move |xs, train| {
        let mut ys = xs.shallow_clone();
        for (attn, ff) in &layers {
            ys += ys.apply_t(attn, train);
            ys += ys.apply_t(ff, train);
        }
        ys
    })
}

pub enum Pool {
    Cls,
    Mean,
}

pub fn vit_3d(
    p: &nn::Path,
    image_size: (i64, i64),
    image_patch_size: (i64, i64),
    frames: i64,
    frame_patch_size: i64,
    num_classes: i64,
    dim: i64,
    depth: i64,
    heads: i64,
    head_dim: i64,
    mlp_dim: i64,
    pool: Pool,
    channels: i64,
    dropout: f64,
    emb_dropout: f64,
) -> impl nn::ModuleT {
    let (h, w) = image_size;
    let (ph, pw) = image_patch_size;
    assert!(
        h % ph == 0 && w % pw == 0,
        "Image dimensions must be divisible by the patch size."
    );
    assert!(
        frames % frame_patch_size == 0,
        "Frames must be divisible by the frame patch size."
    );

    let num_patches = (h / ph) * (w / pw) * (frames / frame_patch_size);
    let patch_dim = channels * ph * pw * frame_patch_size;

    let q = p / "to_patch_embedding";
    let to_patch_embedding = nn::seq_t()
        .add_fn(move |xs| {
            xs.reshape([
                -1,
                channels,
                frames / frame_patch_size,
                frame_patch_size,
                h / ph,
                ph,
                w / pw,
                pw,
            ])
            .permute([0, 2, 4, 6, 5, 7, 3, 1])
            .reshape([
                -1,
                frames / frame_patch_size * h / ph * w / pw,
                ph * pw * frame_patch_size * channels,
            ])
        })
        .add(nn::layer_norm(&q / 1, vec![patch_dim], Default::default()))
        .add(nn::linear(&q / 2, patch_dim, dim, Default::default()))
        .add(nn::layer_norm(&q / 3, vec![dim], Default::default()));
    let pos_embedding = p.var(
        "pos_embedding",
        &[1, num_patches + 1, dim],
        nn::Init::Randn {
            mean: 0.0,
            stdev: 1.0,
        },
    );

    let cls_token = p.var(
        "cls_token",
        &[1, 1, dim],
        nn::Init::Randn {
            mean: 0.0,
            stdev: 1.0,
        },
    );

    let transformer = transformer(
        p / "transformer",
        dim,
        depth,
        heads,
        head_dim,
        mlp_dim,
        dropout,
    );

    let mlp_head = nn::seq_t()
        .add(nn::layer_norm(
            p / "mlp_head" / 0,
            vec![dim],
            Default::default(),
        ))
        .add(nn::linear(
            p / "mlp_head" / 1,
            dim,
            num_classes,
            Default::default(),
        ));
    nn::func_t(move |xs, train| {
        let mut ys = xs.apply_t(&to_patch_embedding, train);
        let (b, n, _) = ys.size3().unwrap();
        let cls_tokens = cls_token.repeat([b, 1, 1]);
        ys = Tensor::cat(&[cls_tokens, ys], 1);
        ys += pos_embedding.i((.., ..(n + 1)));
        ys = ys.dropout(emb_dropout, train).apply_t(&transformer, train);

        ys = match pool {
            Pool::Cls => ys.i((.., 0)),
            Pool::Mean => ys.mean_dim(1, true, tch::Kind::Float),
        };
        ys.apply_t(&mlp_head, train)
    })
}
