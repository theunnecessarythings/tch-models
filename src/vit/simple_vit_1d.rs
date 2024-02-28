use tch::{nn, Kind, Tensor};

fn posemb_sincos_1d(patches: &Tensor, temperature: f64, dtype: Kind) -> Tensor {
    let (_, n, dim) = patches.size3().unwrap();
    let device = patches.device();

    assert!(
        dim % 2 == 0,
        "feature dimension must be multiple of 2 for sincos emb"
    );

    let mut n = Tensor::arange(n, (Kind::Float, device));
    let mut omega = Tensor::arange(dim / 2, (Kind::Float, device)) / (dim / 2 - 1) as f64;
    omega = 1. / Tensor::pow_scalar(temperature, &omega);

    n = n.unsqueeze(-1) * omega.unsqueeze(0);
    Tensor::cat(&[n.sin(), n.cos()], 1).to_kind(dtype)
}

fn feed_forward(p: nn::Path, dim: i64, hidden_dim: i64) -> impl nn::ModuleT {
    let q = &p / "net";
    nn::seq_t()
        .add(nn::layer_norm(&q / 0, vec![dim], Default::default()))
        .add(nn::linear(&q / 1, dim, hidden_dim, Default::default()))
        .add_fn(|xs| xs.gelu("none"))
        .add(nn::linear(&q / 3, hidden_dim, dim, Default::default()))
}

fn attention(p: nn::Path, dim: i64, heads: i64, head_dim: i64) -> impl nn::ModuleT {
    let scale = 1. / (head_dim as f64).sqrt();
    let norm = nn::layer_norm(&p / "norm", vec![dim], Default::default());
    let to_qkv = nn::linear(
        &p / "to_qkv",
        dim,
        head_dim * 3 * heads,
        nn::LinearConfig {
            bias: false,
            ..Default::default()
        },
    );
    let to_out = nn::linear(
        &p / "to_out",
        head_dim * heads,
        dim,
        nn::LinearConfig {
            bias: false,
            ..Default::default()
        },
    );
    nn::func_t(move |xs, _| {
        let qkv = xs
            .apply(&norm)
            .apply(&to_qkv)
            .view([-1, xs.size()[1], heads, 3 * head_dim])
            .transpose(1, 2);
        let (q, k, v) = (&qkv.select(3, 0), &qkv.select(3, 1), &qkv.select(3, 2));
        let dots = q.matmul(&k.transpose(-1, -2)) * scale;
        let attn = dots.softmax(-1, tch::Kind::Float);
        let out = attn.matmul(v);
        let out = out
            .transpose(1, 2)
            .contiguous()
            .view([-1, xs.size()[1], head_dim * heads]);
        out.apply(&to_out)
    })
}

fn transformer(
    p: nn::Path,
    dim: i64,
    depth: i64,
    heads: i64,
    head_dim: i64,
    mlp_dim: i64,
) -> impl nn::ModuleT {
    let norm = nn::layer_norm(&p / "norm", vec![dim], Default::default());
    let mut layers = vec![];
    let q = &p / "layers";
    for i in 0..depth {
        layers.push((
            attention(&q / i / 0, dim, heads, head_dim),
            feed_forward(&q / i / 1, dim, mlp_dim),
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

pub fn simple_vit_1d(
    p: &nn::Path,
    seq_len: i64,
    patch_size: i64,
    num_classes: i64,
    dim: i64,
    depth: i64,
    heads: i64,
    mlp_dim: i64,
    channels: i64,
    head_dim: i64,
) -> impl nn::ModuleT {
    assert!(seq_len % patch_size == 0);

    let num_patches = seq_len / patch_size;
    let patch_dim = channels * patch_size;

    let q = p / "to_patch_embedding";
    let to_patch_embedding = nn::seq()
        .add_fn(move |xs| {
            xs.view([-1, channels, num_patches, patch_size])
                .permute([0, 2, 3, 1])
                .view([-1, num_patches, patch_dim])
        })
        .add(nn::layer_norm(&q / 1, vec![patch_dim], Default::default()))
        .add(nn::linear(&q / 2, patch_dim, dim, Default::default()))
        .add(nn::layer_norm(&q / 3, vec![dim], Default::default()));

    let transformer = transformer(p / "transformer", dim, depth, heads, head_dim, mlp_dim);

    let linear_head = nn::linear(p / "linear_head", dim, num_classes, Default::default());

    nn::func_t(move |xs, train| {
        let size = xs.size();
        let d = size.last().unwrap();
        let b = size[0];
        let mut ys = xs.apply_t(&to_patch_embedding, train);
        let pe = posemb_sincos_1d(&ys, 10000., tch::Kind::Float);
        ys = ys.reshape([b, -1, *d]) + pe;
        ys.apply_t(&transformer, train)
            .mean_dim(1, false, Kind::Float)
            .apply(&linear_head)
    })
}
