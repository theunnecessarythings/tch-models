use tch::{nn, IndexOp, Kind};

fn patch_merger(p: nn::Path, dim: i64, num_tokens_out: i64) -> impl nn::ModuleT {
    let scale = 1. / (dim as f64).sqrt();
    let norm = nn::layer_norm(&p / "norm", vec![dim], Default::default());
    let queries = p.var(
        "queries",
        &[num_tokens_out, dim],
        nn::Init::Randn {
            mean: 0.0,
            stdev: 1.0,
        },
    );

    nn::func_t(move |xs, _| {
        let ys = xs.apply(&norm);
        let sim = queries.matmul(&ys.transpose(-1, -2)) * scale;
        sim.softmax(-1, Kind::Float).matmul(&ys)
    })
}

fn feed_forward(p: nn::Path, dim: i64, hidden_dim: i64, dropout: f64) -> impl nn::ModuleT {
    let q = &p / "net";
    nn::seq_t()
        .add(nn::layer_norm(&q / 0, vec![dim], Default::default()))
        .add(nn::linear(&q / 1, dim, hidden_dim, Default::default()))
        .add_fn_t(move |xs, train| xs.gelu("none").dropout(dropout, train))
        .add(nn::linear(&q / 4, hidden_dim, dim, Default::default()))
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
            .softmax(-1, Kind::Float)
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
    patch_merge_layer: Option<i64>,
    patch_merge_num_tokens: i64,
) -> impl nn::ModuleT {
    let mut layers = vec![];
    let norm = nn::layer_norm(&p / "norm", vec![dim], Default::default());

    let patch_merge_layer_idx = patch_merge_layer.unwrap_or(depth / 2) - 1;
    let patch_merger = patch_merger(&p / "patch_merger", dim, patch_merge_num_tokens);

    let q = &p / "layers";
    for i in 0..depth {
        layers.push((
            attention(&q / i / 0, dim, heads, head_dim, dropout),
            feed_forward(&q / i / 1, dim, mlp_dim, dropout),
        ));
    }
    nn::func_t(move |xs, train| {
        let mut ys = xs.shallow_clone();
        for (idx, (attn, ff)) in layers.iter().enumerate() {
            ys += ys.apply_t(attn, train);
            ys += ys.apply_t(ff, train);

            if idx as i64 == patch_merge_layer_idx {
                ys = ys.apply_t(&patch_merger, train);
            }
        }
        ys.apply(&norm)
    })
}

pub fn vit_with_patch_merger(
    p: &nn::Path,
    image_size: (i64, i64),
    patch_size: (i64, i64),
    num_classes: i64,
    dim: i64,
    depth: i64,
    heads: i64,
    mlp_dim: i64,
    patch_merge_layer: Option<i64>,
    patch_merge_num_tokens: i64,
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
    let patch_dim = channels * ph * pw;

    let q = p / "patch_embedding";
    let to_patch_embedding = nn::seq()
        .add_fn(move |xs| {
            xs.view([-1, xs.size()[1], h / ph, ph, w / pw, pw])
                .permute([0, 2, 4, 3, 5, 1])
                .contiguous()
                .view([-1, (h / ph) * (w / pw), ph * pw * xs.size()[1]])
        })
        .add(nn::layer_norm(&q / 1, vec![patch_dim], Default::default()))
        .add(nn::linear(&q / 2, patch_dim, dim, Default::default()))
        .add(nn::layer_norm(&q / 3, vec![dim], Default::default()));

    let pos_embedding = p.var(
        "pos_embedding",
        &[1, num_patches + 1, dim],
        nn::Init::Randn {
            mean: 0.,
            stdev: 1.,
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
        patch_merge_layer,
        patch_merge_num_tokens,
    );

    let mlp_head = nn::seq()
        .add_fn(|xs| xs.mean_dim(1, false, Kind::Float))
        .add(nn::linear(
            p / "mlp_head" / 1,
            dim,
            num_classes,
            Default::default(),
        ));

    nn::func_t(move |xs, train| {
        let mut ys = xs.apply_t(&to_patch_embedding, train);
        let (_, n, _) = ys.size3().unwrap();
        ys += &pos_embedding.i((0, ..n));
        ys.dropout(emb_dropout, train)
            .apply_t(&transformer, train)
            .apply(&mlp_head)
    })
}
