use tch::{nn, IndexOp, Tensor};

fn feed_forward(p: nn::Path, dim: i64, hidden_dim: i64, dropout: f64) -> impl nn::ModuleT {
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
        inner_dim,
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
            .reshape([-1, out.size()[1], inner_dim])
            .apply_t(&to_out, train)
    })
}

pub enum Pool {
    Cls,
    Mean,
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
    let norm = nn::layer_norm(&p / "norm", vec![dim], Default::default());
    let mut layers = vec![];
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
        ys.apply(&norm)
    })
}

pub fn vivit(
    p: &nn::Path,
    image_size: (i64, i64),
    image_patch_size: (i64, i64),
    frames: i64,
    frame_patch_size: i64,
    num_classes: i64,
    dim: i64,
    spatial_depth: i64,
    temporal_depth: i64,
    heads: i64,
    mlp_dim: i64,
    pool: Pool,
    channels: i64,
    head_dim: i64,
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
        "Number of frames must be divisible by the frame patch size."
    );
    let num_image_patches = (h / ph) * (w / pw);
    let num_frame_patches = frames / frame_patch_size;
    let patch_dim = channels * ph * pw * frame_patch_size;

    let q = p / "to_patch_embedding";
    let to_patch_embedding = nn::seq_t()
        .add_fn(move |xs| {
            let (b, c, f, h, w) = xs.size5().unwrap();
            let pf = frame_patch_size;
            xs.reshape([b, c, f / pf, pf, h / ph, ph, w / pw, pw])
                .permute([0, 2, 4, 6, 5, 7, 3, 1])
                .contiguous()
                .view([b, f / pf, (h / ph) * (w / pw), ph * pw * pf * c])
        })
        .add(nn::layer_norm(&q / 1, vec![patch_dim], Default::default()))
        .add(nn::linear(&q / 2, patch_dim, dim, Default::default()))
        .add(nn::layer_norm(&q / 3, vec![dim], Default::default()));

    let pos_embedding = p.var(
        "pos_embedding",
        &[1, num_frame_patches, num_image_patches, dim],
        nn::Init::Randn {
            mean: 0.0,
            stdev: 1.0,
        },
    );

    let (spatial_cls_token, temporal_cls_token) = match pool {
        Pool::Cls => (
            Some(p.var(
                "spatial_cls_token",
                &[1, 1, dim],
                nn::Init::Randn {
                    mean: 0.0,
                    stdev: 1.0,
                },
            )),
            Some(p.var(
                "temporal_cls_token",
                &[1, 1, dim],
                nn::Init::Randn {
                    mean: 0.0,
                    stdev: 1.0,
                },
            )),
        ),
        Pool::Mean => (None, None),
    };

    let spatial_transformer = transformer(
        p / "spatial_transformer",
        dim,
        spatial_depth,
        heads,
        head_dim,
        mlp_dim,
        dropout,
    );
    let temporal_transformer = transformer(
        p / "temporal_transformer",
        dim,
        temporal_depth,
        heads,
        head_dim,
        mlp_dim,
        dropout,
    );

    let mlp_head = nn::linear(p / "mlp_head", dim, num_classes, Default::default());

    nn::func_t(move |xs, train| {
        let mut ys = xs.apply_t(&to_patch_embedding, train);
        let (b, f, n, _) = ys.size4().unwrap();

        ys += pos_embedding.i((.., ..f, ..n, ..));
        if let Some(spatial_cls_token) = &spatial_cls_token {
            ys = Tensor::cat(&[spatial_cls_token.repeat([b, f, 1, 1]), ys], 2);
        }

        ys = ys
            .dropout(emb_dropout, train)
            .reshape([-1, n, dim])
            .apply_t(&spatial_transformer, train)
            .reshape([b, f, n, dim]);

        ys = match pool {
            Pool::Cls => ys.i((.., .., 0)),
            Pool::Mean => ys.mean_dim(2, false, tch::Kind::Float),
        };
        if let Some(temporal_cls_token) = &temporal_cls_token {
            ys = Tensor::cat(
                &[temporal_cls_token.repeat([b, 1, 1]), ys.i((.., .., 0))],
                1,
            );
        }

        ys = ys.apply_t(&temporal_transformer, train);

        ys = match pool {
            Pool::Cls => ys.i((.., 0)),
            Pool::Mean => ys.mean_dim(1, false, tch::Kind::Float),
        };
        ys.apply(&mlp_head)
    })
}