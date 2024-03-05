use tch::{nn, Kind, Tensor};

fn patch_dropout_block(prob: f64) -> impl nn::ModuleT {
    nn::func_t(move |xs, train| {
        if !train || prob == 0. {
            return xs.shallow_clone();
        }
        let (b, n, _) = xs.size3().unwrap();
        let batch_indices = Tensor::arange(b, (Kind::Int64, xs.device())).unsqueeze(-1);
        let num_patches_keep = i64::max(1, ((1. - prob) * n as f64) as i64);
        let patch_indices_keep = Tensor::randn([b, n], (Kind::Float, xs.device()))
            .topk(num_patches_keep, -1, true, true)
            .1;
        xs.index_select(0, &batch_indices)
            .index_select(1, &patch_indices_keep)
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
) -> impl nn::ModuleT {
    let mut layers = vec![];
    let q = &p / "layers";
    for i in 0..depth {
        layers.push((
            attention(&q / i / 0, dim, heads, head_dim, dropout),
            feed_forward(&q / i / 1, dim, mlp_dim, dropout),
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

pub fn vit_with_patch_dropout(
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
    patch_dropout: f64,
) -> impl nn::ModuleT {
    let (h, w) = image_size;
    let (ph, pw) = patch_size;
    assert!(
        h % ph == 0 && w % pw == 0,
        "Image dimensions must be divisible by the patch size.",
    );

    let num_patches = (h / ph) * (w / pw);
    let patch_dim = channels * ph * pw;

    let q = p / "to_patch_embedding";
    let to_patch_embedding = nn::seq()
        .add_fn(move |xs| {
            xs.view([-1, xs.size()[1], h / ph, ph, w / pw, pw])
                .permute([0, 2, 4, 3, 5, 1])
                .contiguous()
                .view([-1, (h / ph) * (w / pw), patch_dim])
        })
        .add(nn::linear(&q / 1, patch_dim, dim, Default::default()));
    let pos_embedding = p.var(
        "pos_embedding",
        &[num_patches, dim],
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
    let patch_dropout = patch_dropout_block(patch_dropout);

    let transformer = transformer(
        p / "transformer",
        dim,
        depth,
        heads,
        head_dim,
        mlp_dim,
        dropout,
    );

    let q = p / "mlp_head";
    let mlp_head = nn::seq()
        .add(nn::layer_norm(&q / 0, vec![dim], Default::default()))
        .add(nn::linear(&q / 1, dim, num_classes, Default::default()));

    nn::func_t(move |xs, train| {
        let mut ys = xs.apply_t(&to_patch_embedding, train);
        let (b, _, _) = ys.size3().unwrap();
        ys += &pos_embedding;
        ys = ys.apply_t(&patch_dropout, train);

        let cls_tokens = cls_token.repeat([b, 1, 1]);
        ys = Tensor::cat(&[cls_tokens, ys], 1)
            .dropout(emb_dropout, train)
            .apply_t(&transformer, train);
        match pool {
            Pool::Cls => ys.select(1, 0),
            Pool::Mean => ys.mean_dim(1, false, Kind::Float),
        }
        .apply(&mlp_head)
    })
}