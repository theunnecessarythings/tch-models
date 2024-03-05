use tch::{nn, IndexOp, Kind, Tensor};

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

pub fn simple_vit_with_fft(
    p: &nn::Path,
    image_size: (i64, i64),
    patch_size: (i64, i64),
    freq_patch_size: (i64, i64),
    num_classes: i64,
    dim: i64,
    depth: i64,
    heads: i64,
    mlp_dim: i64,
    channels: i64,
    head_dim: i64,
) -> impl nn::ModuleT {
    let (h, w) = image_size;
    let (ph, pw) = patch_size;
    let (fh, fw) = freq_patch_size;

    assert!(
        h % ph == 0 && w % pw == 0,
        "image dimensions must be divisible by patch size"
    );
    assert!(
        h % fh == 0 && w % fw == 0,
        "image dimensions must be divisible by freq patch size"
    );

    let patch_dim = ph * pw * channels;
    let freq_patch_dim = fh * fw * channels * 2;

    let q = p / "to_patch_embedding";
    let to_patch_embedding = nn::seq()
        .add_fn(move |xs| {
            xs.view([-1, channels, h / ph, ph, w / pw, pw])
                .permute(&[0, 2, 4, 3, 5, 1])
                .contiguous()
                .view([-1, (h / ph) * (w / pw), patch_dim])
        })
        .add(nn::layer_norm(&q / 1, vec![patch_dim], Default::default()))
        .add(nn::linear(&q / 2, patch_dim, dim, Default::default()))
        .add(nn::layer_norm(&q / 3, vec![dim], Default::default()));

    let q = p / "to_freq_embedding";
    let to_freq_embedding = nn::seq()
        .add_fn(move |xs| {
            xs.reshape([-1, channels, h / fh, fh, w / fw, fw, 2])
                .permute(&[0, 2, 4, 3, 5, 6, 1])
                .contiguous()
                .view([-1, (h / fh) * (w / fw), freq_patch_dim])
        })
        .add(nn::layer_norm(
            &q / 1,
            vec![freq_patch_dim],
            Default::default(),
        ))
        .add(nn::linear(&q / 2, freq_patch_dim, dim, Default::default()))
        .add(nn::layer_norm(&q / 3, vec![dim], Default::default()));

    let pos_embedding = posemb_sincos_2d(h / ph, w / pw, dim, 10000, Kind::Float);
    let freq_pos_embedding = posemb_sincos_2d(h / fh, w / fw, dim, 10000, Kind::Float);

    let transformer = transformer(p / "transformer", dim, depth, heads, head_dim, mlp_dim);

    let linear_head = nn::linear(p / "linear_head", dim, num_classes, Default::default());

    nn::func_t(move |xs, train| {
        let mut ys = xs.apply(&to_patch_embedding);
        let size = ys.size();
        let dim = ys.dim();
        let mut fs = xs
            .fft_fft2(&[size[dim - 2], size[dim - 1]][..], [-2, -1], "backward")
            .view_as_real()
            .apply(&to_freq_embedding);
        ys += &pos_embedding;
        fs += &freq_pos_embedding;

        Tensor::cat(&[fs, ys], -1)
            .apply_t(&transformer, train)
            .i((.., -1))
            .mean_dim(1, false, Kind::Float)
            .apply(&linear_head)
    })
}