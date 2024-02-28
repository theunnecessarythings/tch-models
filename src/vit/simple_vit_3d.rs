use tch::{nn, Kind, Tensor};

fn posemb_sincos_3d(patches: &Tensor, temperature: i64) -> Tensor {
    let (_, f, h, w, dim) = patches.size5().unwrap();
    let device = patches.device();
    let kind = patches.kind();

    let zyx = Tensor::meshgrid_indexing(
        &[
            &Tensor::arange(f, (Kind::Int64, device)),
            &Tensor::arange(h, (Kind::Int64, device)),
            &Tensor::arange(w, (Kind::Int64, device)),
        ],
        "ij",
    );
    let (z, y, x) = (&zyx[0], &zyx[1], &zyx[2]);
    let fourier_dim = dim / 6;
    let mut omega = Tensor::arange(fourier_dim, (Kind::Float, device)) / (fourier_dim - 1) as f64;
    omega = 1. / Tensor::pow_scalar(temperature as f64, &omega);

    let z = z.unsqueeze(-1) * omega.unsqueeze(0);
    let y = y.unsqueeze(-1) * omega.unsqueeze(0);
    let x = x.unsqueeze(-1) * omega.unsqueeze(0);

    Tensor::cat(
        &[&x.sin(), &x.cos(), &y.sin(), &y.cos(), &z.sin(), &z.cos()],
        1,
    )
    .pad([0, dim - (fourier_dim * 6)], "constant", None)
    .to_kind(kind)
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

pub fn simple_vit_3d(
    p: &nn::Path,
    image_size: (i64, i64),
    image_patch_size: (i64, i64),
    frames: i64,
    frame_patch_size: i64,
    num_classes: i64,
    dim: i64,
    depth: i64,
    heads: i64,
    mlp_dim: i64,
    channels: i64,
    head_dim: i64,
) -> impl nn::ModuleT {
    let (h, w) = image_size;
    let (ph, pw) = image_patch_size;

    assert!(
        h % ph == 0 && w % pw == 0,
        "Image dimensions must be divisible by the patch size."
    );
    assert!(
        frames % frame_patch_size == 0,
        "Frames must be divisible by the patch size."
    );

    let patch_dim = channels * ph * pw * frame_patch_size;

    let q = p / "to_patch_embedding";
    let to_patch_embedding = nn::seq()
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

    let transformer = transformer(p / "transformer", dim, depth, heads, head_dim, mlp_dim);

    let linear_head = nn::linear(p / "linear_head", dim, num_classes, Default::default());

    nn::func_t(move |xs, train| {
        let mut ys = xs.apply_t(&to_patch_embedding, train);
        let size = ys.size();
        let (b, d) = (size[0], size.last().unwrap());
        ys = ys.view([b, -1, *d]) + posemb_sincos_3d(&ys, 10000);
        ys.apply_t(&transformer, train)
            .mean_dim(1, false, Kind::Float)
            .apply(&linear_head)
    })
}
