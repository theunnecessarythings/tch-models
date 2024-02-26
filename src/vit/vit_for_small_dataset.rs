use tch::{nn, Kind, Tensor};

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

// fn spt(p: nn::Path, dim: i64, patch_size: i64, channels: i64) -> impl nn::ModuleT {
//     let patch_dim = patch_size * patch_size * channels * 5;
//     let to_patch_tokens = nn::seq()
//         .add_fn(move|xs| {
//             let [b, c, h, w] = xs.size4().unwrap();
//         })
// }
