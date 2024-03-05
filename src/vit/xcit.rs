use tch::{nn, Tensor};

fn layer_scale(p: nn::Path, dim: i64, block: Box<dyn nn::ModuleT>, depth: i64) -> impl nn::ModuleT {
    let init_eps = match depth {
        _ if depth <= 18 => 0.1,
        _ if depth > 18 && depth <= 24 => 1e-5,
        _ => 1e-6,
    };
    let scale = p.var("scale", &[dim], nn::Init::Const(init_eps));

    nn::func_t(move |xs, train| block.forward_t(xs, train) * &scale)
}

fn feed_forward(p: nn::Path, dim: i64, hidden_dim: i64, dropout: f64) -> impl nn::ModuleT {
    nn::seq_t()
        .add(nn::layer_norm(&p / 0, vec![dim], Default::default()))
        .add(nn::linear(&p / 1, dim, hidden_dim, Default::default()))
        .add_fn_t(move |xs, train| xs.gelu("none").dropout(dropout, train))
        .add(nn::linear(&p / 4, hidden_dim, dim, Default::default()))
        .add_fn_t(move |xs, train| xs.dropout(dropout, train))
}

fn attention(
    p: nn::Path,
    dim: i64,
    heads: i64,
    head_dim: i64,
    dropout: f64,
) -> impl Fn(&Tensor, bool, Option<&Tensor>) -> Tensor {
    let inner_dim = head_dim * heads;
    let scale = 1. / (head_dim as f64).sqrt();

    let norm = nn::layer_norm(&p / "norm", vec![dim], Default::default());
    let to_q = nn::linear(
        &p / "to_q",
        dim,
        inner_dim,
        nn::LinearConfig {
            bias: false,
            ..Default::default()
        },
    );
    let to_kv = nn::linear(
        &p / "to_kv",
        dim,
        inner_dim * 2,
        nn::LinearConfig {
            bias: false,
            ..Default::default()
        },
    );

    let to_out = nn::seq_t()
        .add(nn::linear(
            &p / "to_out" / 0,
            inner_dim,
            dim,
            Default::default(),
        ))
        .add_fn_t(move |xs, train| xs.dropout(dropout, train));

    move |xs, train, context| {
        let ys = xs.apply(&norm);
        let context = match context {
            Some(context) => Tensor::cat(&[&ys, context], 1),
            None => ys.shallow_clone(),
        };

        let q = ys.apply(&to_q);
        let kv = context.apply(&to_kv).chunk(2, -1);
        let (k, v) = (&kv[0], &kv[1]);

        let qkv = [&q, k, v]
            .iter()
            .map(|x| x.view([x.size()[0], -1, heads, head_dim]).transpose(1, 2))
            .collect::<Vec<_>>();
        let (q, k, v) = (&qkv[0], &qkv[1], &qkv[2]);

        let attn = q.matmul(&k.transpose(-2, -1)) * scale;
        let attn = attn.softmax(-1, tch::Kind::Float).dropout(dropout, train);
        attn.matmul(v)
            .transpose(1, 2)
            .view([ys.size()[0], -1, inner_dim])
            .apply_t(&to_out, train)
    }
}

fn normalize(x: &Tensor, p: f64, dim: i64, eps: f64) -> Tensor {
    let norm = x
        .norm_scalaropt_dim(p, dim, true)
        .clamp_min(eps)
        .expand_as(x);
    x / norm
}

fn xc_attention(
    p: nn::Path,
    dim: i64,
    heads: i64,
    head_dim: i64,
    dropout: f64,
) -> impl nn::ModuleT {
    let inner_dim = head_dim * heads;
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

    let temperature = p.var("temperature", &[heads, 1, 1], nn::Init::Const(1.0));

    let to_out = nn::seq_t()
        .add(nn::linear(
            &p / "to_out" / 0,
            inner_dim,
            dim,
            Default::default(),
        ))
        .add_fn_t(move |xs, train| xs.dropout(dropout, train));

    nn::func_t(move |xs, train| {
        let size = xs.size();
        let (b, d) = (size[0], size.last().unwrap());
        let ys = xs.view([b, -1, *d]).apply(&norm);
        let qkv = ys
            .apply(&to_qkv)
            .chunk(3, -1)
            .iter()
            .map(|x| x.view([b, -1, heads, head_dim]).permute([0, 2, 3, 1]))
            .collect::<Vec<_>>();
        let (q, k, v) = (&qkv[0], &qkv[1], &qkv[2]);
        let q = normalize(q, 2., -1, 1e-12);
        let k = normalize(k, 2., -1, 1e-12);

        let attn = q.matmul(&k.transpose(-2, -1)) * temperature.exp();
        let attn = attn.softmax(-1, tch::Kind::Float).dropout(dropout, train);

        attn.matmul(v)
            .permute([0, 3, 1, 2])
            .view(size.as_ref())
            .apply_t(&to_out, train)
    })
}

fn local_patch_interaction(p: nn::Path, dim: i64, ksize: i64) -> impl nn::ModuleT {
    assert!(ksize % 2 == 1);

    let pad = ksize / 2;

    nn::seq_t()
        .add(nn::layer_norm(&p / 0, vec![dim], Default::default()))
        .add_fn(|xs| xs.permute([0, 3, 1, 2]))
        .add(nn::conv2d(
            &p / 2,
            dim,
            dim,
            ksize,
            nn::ConvConfig {
                padding: pad,
                groups: dim,
                ..Default::default()
            },
        ))
        .add(nn::batch_norm2d(&p / 3, dim, Default::default()))
        .add_fn(|xs| xs.gelu("none"))
        .add(nn::conv2d(
            &p / 5,
            dim,
            dim,
            ksize,
            nn::ConvConfig {
                padding: pad,
                groups: dim,
                ..Default::default()
            },
        ))
        .add_fn(|xs| xs.permute(&[0, 2, 3, 1]))
}

fn transformer(
    p: nn::Path,
    dim: i64,
    depth: i64,
    heads: i64,
    head_dim: i64,
    mlp_dim: i64,
    dropout: f64,
    layer_dropout: f64,
) -> impl nn::ModuleT {
}
