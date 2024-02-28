use tch::{nn, Kind, Tensor};

fn posemb_sincos_2d(h: i64, w: i64, dim: i64, temperature: i64, dtype: Kind) -> Tensor {
    let yx = Tensor::meshgrid_indexing(
        &[
            &Tensor::arange(h, (Kind::Int64, tch::Device::Cpu)),
            &Tensor::arange(w, (Kind::Int64, tch::Device::Cpu)),
        ],
        "ij",
    );
    let (y, x) = (&yx[0], &yx[1]);
    assert!(
        dim % 4 == 0,
        "feature dimension must be multiple of 4 for sincos emb"
    );
    let mut omega = Tensor::arange(dim / 4, (Kind::Float, tch::Device::Cpu)) / (dim / 4 - 1) as f64;
    omega = 1. / Tensor::pow_scalar(temperature as f64, &omega);

    let y = y.unsqueeze(-1) * omega.unsqueeze(0);
    let x = x.unsqueeze(-1) * omega.unsqueeze(0);
    Tensor::cat(&[&x.sin(), &x.cos(), &y.sin(), &y.cos()], 1).to_kind(dtype)
}

fn normalize(x: &Tensor, p: f64, dim: i64, eps: f64) -> Tensor {
    let norm = x
        .norm_scalaropt_dim(p, dim, true)
        .clamp_min(eps)
        .expand_as(x);
    x / norm
}

fn rms_norm(p: nn::Path, heads: i64, dim: i64) -> impl nn::ModuleT {
    let scale = 1. / (dim as f64).sqrt();
    let gamma = p.var("gamma", &[heads, 1, dim], nn::Init::Const(1. / scale));

    nn::func_t(move |xs, _| {
        let norm = normalize(xs, 2.0, -1, 1e-6);
        norm * scale * &gamma
    })
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

    let q_norm = rms_norm(&p / "q_norm", heads, head_dim);
    let k_norm = rms_norm(&p / "k_norm", heads, head_dim);

    nn::func_t(move |xs, _| {
        let qkv = xs
            .apply(&norm)
            .apply(&to_qkv)
            .view([-1, xs.size()[1], heads, 3 * head_dim])
            .transpose(1, 2);
        let (q, k, v) = (&qkv.select(3, 0), &qkv.select(3, 1), &qkv.select(3, 2));
        let q = q.apply_t(&q_norm, false);
        let k = k.apply_t(&k_norm, false);
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

pub fn simple_vit_with_qk_norm(
    p: &nn::Path,
    image_size: (i64, i64),
    patch_size: (i64, i64),
    _num_classes: i64,
    dim: i64,
    depth: i64,
    heads: i64,
    mlp_dim: i64,
    channels: i64,
    head_dim: i64,
) -> impl nn::ModuleT {
    let (h, w) = image_size;
    let (ph, pw) = patch_size;
    assert!(
        h % ph == 0 && w % pw == 0,
        "Image dimensions must be divisible by the patch size."
    );

    let patch_dim = channels * ph * pw;

    let q = p / "to_patch_embedding";
    let to_patch_embedding = nn::seq()
        .add_fn(move |xs| {
            xs.view([-1, xs.size()[1], h / ph, ph, w / pw, pw])
                .permute([0, 2, 4, 3, 5, 1])
                .contiguous()
                .view([-1, (h / ph) * (w / pw), patch_dim])
        })
        .add(nn::layer_norm(&q / 1, vec![patch_dim], Default::default()))
        .add(nn::linear(&q / 2, patch_dim, dim, Default::default()))
        .add(nn::layer_norm(&q / 3, vec![dim], Default::default()));

    let pos_embedding = posemb_sincos_2d(h / ph, w / pw, dim, 10000, Kind::Float);

    let transformer = transformer(p / "transformer", dim, depth, heads, head_dim, mlp_dim);

    let linear_head = nn::layer_norm(p / "linear_head", vec![dim], Default::default());

    nn::func_t(move |xs, train| {
        let mut ys = xs.apply(&to_patch_embedding);
        ys += &pos_embedding;
        ys.apply_t(&transformer, train)
            .mean_dim(1, false, Kind::Float)
            .apply(&linear_head)
    })
}
