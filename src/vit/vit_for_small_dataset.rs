use tch::{nn, IndexOp, Kind, Tensor};

fn feed_forward(p: nn::Path, dim: i64, hidden_dim: i64, dropout: f64) -> impl nn::ModuleT {
    nn::seq_t()
        .add(nn::layer_norm(&p / 0, vec![dim], Default::default()))
        .add(nn::linear(&p / 1, dim, hidden_dim, Default::default()))
        .add_fn_t(move |xs, train| xs.gelu("none").dropout(dropout, train))
        .add(nn::linear(&p / 4, hidden_dim, dim, Default::default()))
        .add_fn_t(move |xs, train| xs.dropout(dropout, train))
}

fn lsa(p: nn::Path, dim: i64, heads: i64, head_dim: i64, dropout: f64) -> impl nn::ModuleT {
    let inner_dim = head_dim * heads;
    let temperature = p.var(
        "temperature",
        &[1],
        nn::Init::Const((1. / (head_dim as f64).sqrt()).ln()),
    );
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
    let to_out = nn::seq()
        .add(nn::linear(
            &p / "to_out" / 0,
            inner_dim,
            dim,
            Default::default(),
        ))
        .add_fn(move |xs| xs.dropout(dropout, true));

    nn::func_t(move |xs, train| {
        let ys = xs.apply(&norm);
        let qkv = ys
            .apply(&to_qkv)
            .chunk(3, -1)
            .iter()
            .map(|x| {
                x.reshape([-1, x.size()[1], heads, head_dim])
                    .transpose(1, 2)
            })
            .collect::<Vec<_>>();
        let (q, k, v) = (&qkv[0], &qkv[1], &qkv[2]);
        let dots = q.matmul(&k.transpose(-1, -2)) * temperature.exp();

        let mask = Tensor::eye(*dots.size().last().unwrap(), (Kind::Bool, dots.device()));
        let dots = dots.masked_fill(&mask, f64::NEG_INFINITY);
        dots.softmax(-1, Kind::Float)
            .dropout(dropout, train)
            .matmul(v)
            .transpose(1, 2)
            .reshape([-1, ys.size()[1], inner_dim])
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
    for i in 0..depth {
        layers.push((
            lsa(&p / i / 0, dim, heads, head_dim, dropout),
            feed_forward(&p / i / 1, dim, mlp_dim, dropout),
        ));
    }

    nn::func_t(move |xs, train| {
        let mut ys = xs.shallow_clone();
        for (attention, feed_forward) in &layers {
            ys += ys.apply_t(attention, train);
            ys += ys.apply_t(feed_forward, train);
        }
        ys
    })
}

fn spt(p: nn::Path, dim: i64, patch_size: i64, channels: i64) -> impl nn::ModuleT {
    let patch_dim = patch_size * patch_size * channels * 5;
    let q = &p / "to_patch_tokens";
    let to_patch_tokens = nn::seq()
        .add_fn(move |xs| {
            let (b, c, h, w) = xs.size4().unwrap();
            xs.view([b, c, h / patch_size, patch_size, w / patch_size, patch_size])
                .permute([0, 2, 4, 3, 5, 1])
                .reshape([
                    b,
                    h / patch_size * w / patch_size,
                    patch_size * patch_size * c,
                ])
        })
        .add(nn::layer_norm(&q / 1, vec![patch_dim], Default::default()))
        .add(nn::linear(&q / 2, patch_dim, dim, Default::default()));

    nn::func_t(move |xs, train| {
        let shifts = [[1, -1, 0, 0], [-1, 1, 0, 0], [0, 0, 1, -1], [0, 0, -1, 1]];
        let shifted_x = shifts
            .iter()
            .map(|shift| xs.pad(shift, "constant", None))
            .collect::<Vec<_>>();
        let x_with_shifts = Tensor::cat(
            &[
                xs,
                &shifted_x[0],
                &shifted_x[1],
                &shifted_x[2],
                &shifted_x[3],
            ],
            1,
        );
        x_with_shifts.apply_t(&to_patch_tokens, train)
    })
}

pub enum Pool {
    Cls,
    Mean,
}

pub fn vit_for_small_dataset(
    p: &nn::Path,
    image_size: (i64, i64),
    patch_size: (i64, i64),
    num_classes: i64,
    dim: i64,
    depth: i64,
    heads: i64,
    mlp_dim: i64,
    pool: Pool,
    channels: i64,
    head_dim: i64,
    dropout: f64,
    emb_dropout: f64,
) -> impl nn::ModuleT {
    let (h, w) = image_size;
    let (ph, pw) = patch_size;
    assert!(
        h % ph == 0 && w % pw == 0,
        "Image dimensions must be divisible by the patch size."
    );

    let num_patches = (h / ph) * (w / pw);

    let to_patch_embedding = spt(p / "to_patch_embedding", dim, ph, channels);

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
        ys = Tensor::cat(&[cls_tokens, ys], 1) + pos_embedding.i((.., ..(n + 1)));
        ys = ys.dropout(emb_dropout, train).apply_t(&transformer, train);

        match pool {
            Pool::Cls => ys.i((.., 0)),
            Pool::Mean => ys.mean_dim(1, false, Kind::Float),
        }
        .apply_t(&mlp_head, train)
    })
}
