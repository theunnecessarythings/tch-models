use tch::{
    nn::{self, ConvConfig},
    Kind,
};

fn layer_norm(p: nn::Path, dim: i64, eps: f64) -> impl nn::ModuleT {
    let g = p.var("g", &[1, dim, 1, 1], nn::Init::Const(1.));
    let b = p.var("b", &[1, dim, 1, 1], nn::Init::Const(0.));

    nn::func_t(move |xs, _| {
        let var = xs.var_dim(1, false, true);
        let mean = xs.mean_dim(1, true, Kind::Float);
        (xs - mean) / (var + eps).sqrt() * &g + &b
    })
}

fn feed_forward(p: nn::Path, dim: i64, mult: i64, dropout: f64) -> impl nn::ModuleT {
    nn::seq_t()
        .add(layer_norm(&p / 0, dim, 1e-5))
        .add(nn::conv2d(&p / 1, dim, dim * mult, 1, Default::default()))
        .add_fn_t(move |xs, train| xs.gelu("none").dropout(dropout, train))
        .add(nn::conv2d(&p / 4, dim * mult, dim, 1, Default::default()))
        .add_fn_t(move |xs, train| xs.dropout(dropout, train))
}

fn patch_embedding(p: nn::Path, dim: i64, dim_out: i64, patch_size: i64) -> impl nn::ModuleT {
    let proj = nn::seq_t()
        .add(layer_norm(&p / 0, patch_size.pow(2) * dim, 1e-5))
        .add(nn::conv2d(
            &p / 1,
            patch_size.pow(2) * dim,
            dim_out,
            1,
            Default::default(),
        ))
        .add(layer_norm(&p / 2, dim_out, 1e-5));
    nn::func_t(move |xs, train| {
        let (b, c, h, w) = xs.size4().unwrap();
        xs.view([b, c, h / patch_size, patch_size, w / patch_size, patch_size])
            .permute([0, 1, 3, 5, 2, 4])
            .contiguous()
            .view([b, c * patch_size.pow(2), h, w])
            .apply_t(&proj, train)
    })
}

fn peg(p: nn::Path, dim: i64, ksize: i64) -> impl nn::ModuleT {
    let proj = nn::conv2d(
        &p / "fn" / "proj",
        dim,
        dim,
        ksize,
        ConvConfig {
            padding: ksize / 2,
            groups: dim,
            stride: 1,
            ..Default::default()
        },
    );
    nn::func_t(move |xs, _| xs + xs.apply(&proj))
}

fn local_attention(
    p: nn::Path,
    dim: i64,
    heads: i64,
    head_dim: i64,
    dropout: f64,
    patch_size: i64,
) -> impl nn::ModuleT {
    let inner_dim = head_dim * heads;
    let scale = 1. / (head_dim as f64).sqrt();

    let norm = layer_norm(&p / "norm", dim, 1e-5);
    let to_q = nn::conv2d(
        &p / "to_q",
        dim,
        inner_dim,
        1,
        ConvConfig {
            bias: false,
            ..Default::default()
        },
    );
    let to_kv = nn::conv2d(
        &p / "to_kv",
        dim,
        inner_dim * 2,
        1,
        ConvConfig {
            bias: false,
            ..Default::default()
        },
    );

    let to_out = nn::seq_t()
        .add(nn::conv2d(
            &p / "to_out" / 0,
            inner_dim,
            dim,
            1,
            Default::default(),
        ))
        .add_fn_t(move |xs, train| xs.dropout(dropout, train));

    nn::func_t(move |xs, train| {
        let fmap = xs.apply_t(&norm, train);
        let (b, n, x, y) = fmap.size4().unwrap();
        let (x, y) = (x / patch_size, y / patch_size);
        let fmap = fmap
            .view([b, n, x, patch_size, y, patch_size])
            .permute([0, 2, 4, 1, 3, 5])
            .contiguous()
            .view([b * x * y, n, patch_size, patch_size]);
        let q = fmap.apply(&to_q);
        let kv = fmap.apply(&to_kv).chunk(2, 1);
        let (k, v) = (&kv[0], &kv[1]);
        let qkv = [&q, k, v]
            .iter()
            .map(|x| {
                x.view([b, heads, q.size()[1] / heads, patch_size, patch_size])
                    .permute([0, 1, 3, 4, 2])
                    .contiguous()
                    .view([-1, patch_size.pow(2), q.size()[1] / heads])
            })
            .collect::<Vec<_>>();
        let (q, k, v) = (&qkv[0], &qkv[1], &qkv[2]);
        let dots = q.matmul(&k.transpose(-1, -2)) * scale;
        let attn = dots.softmax(-1, Kind::Float);
        attn.matmul(v)
            .reshape([b, x, y, heads, patch_size, patch_size, -1])
            .permute([0, 3, 6, 1, 4, 2, 5])
            .contiguous()
            .view([b, -1, x * patch_size, y * patch_size])
            .apply_t(&to_out, train)
    })
}

fn global_attention(
    p: nn::Path,
    dim: i64,
    heads: i64,
    head_dim: i64,
    dropout: f64,
    k: i64,
) -> impl nn::ModuleT {
    let inner_dim = head_dim * heads;
    let scale = 1. / (head_dim as f64).sqrt();

    let norm = layer_norm(&p / "norm", dim, 1e-5);
    let to_q = nn::conv2d(
        &p / "to_q",
        dim,
        inner_dim,
        1,
        ConvConfig {
            bias: false,
            ..Default::default()
        },
    );
    let to_kv = nn::conv2d(
        &p / "to_kv",
        dim,
        inner_dim * 2,
        k,
        ConvConfig {
            stride: k,
            bias: false,
            ..Default::default()
        },
    );

    let to_out = nn::seq_t()
        .add(nn::conv2d(
            &p / "to_out" / 0,
            inner_dim,
            dim,
            1,
            Default::default(),
        ))
        .add_fn_t(move |xs, train| xs.dropout(dropout, train));

    nn::func_t(move |xs, train| {
        let ys = xs.apply_t(&norm, train);
        let (b, _, x, y) = ys.size4().unwrap();
        let q = ys.apply(&to_q);
        let kv = ys.apply(&to_kv).chunk(2, 1);
        let (k, v) = (&kv[0], &kv[1]);

        let qkv = [&q, k, v]
            .iter()
            .map(|x| {
                x.view([b, heads, q.size()[1] / heads, x.size()[2], x.size()[3]])
                    .permute([0, 1, 3, 4, 2])
                    .contiguous()
                    .view([-1, x.size()[2] * x.size()[3], q.size()[1] / heads])
            })
            .collect::<Vec<_>>();
        let (q, k, v) = (&qkv[0], &qkv[1], &qkv[2]);

        let dots = q.matmul(&k.transpose(-1, -2)) * scale;
        let attn = dots.softmax(-1, Kind::Float).dropout(dropout, train);
        attn.matmul(v)
            .view([b, heads, x, y, -1])
            .permute([0, 4, 1, 2, 3])
            .contiguous()
            .view([b, -1, x, y])
            .apply_t(&to_out, train)
    })
}

fn residual(block: Box<dyn nn::ModuleT>) -> impl nn::ModuleT {
    nn::func_t(move |xs, train| xs + block.forward_t(xs, train))
}

fn transformer(
    p: nn::Path,
    dim: i64,
    depth: i64,
    heads: i64,
    head_dim: i64,
    mlp_mult: i64,
    local_patch_size: i64,
    global_k: i64,
    dropout: f64,
    has_local: bool,
) -> impl nn::ModuleT {
    let mut layers = nn::seq_t();
    for i in 0..depth {
        let q = &p / i;
        let mut block = nn::seq_t();
        if has_local {
            block = block
                .add(residual(Box::new(local_attention(
                    &q / 0 / "fn",
                    dim,
                    heads,
                    head_dim,
                    dropout,
                    local_patch_size,
                ))))
                .add(residual(Box::new(feed_forward(
                    &q / 1 / "fn",
                    dim,
                    mlp_mult,
                    dropout,
                ))));
        }
        block = block
            .add(residual(Box::new(global_attention(
                &q / if has_local { 2 } else { 0 } / "fn",
                dim,
                heads,
                head_dim,
                dropout,
                global_k,
            ))))
            .add(residual(Box::new(feed_forward(
                &q / if has_local { 3 } else { 1 } / "fn",
                dim,
                mlp_mult,
                dropout,
            ))));

        layers = layers.add(block);
    }

    nn::func_t(move |xs, train| xs.apply_t(&layers, train))
}

#[derive(Debug, Clone)]
pub struct BlockConfig {
    emb_dim: i64,
    patch_size: i64,
    local_patch_size: i64,
    global_k: i64,
    depth: i64,
}

pub fn twins_svt(
    p: &nn::Path,
    num_classes: i64,
    block_configs: &[BlockConfig; 4],
    peg_kernel_size: i64,
    dropout: f64,
) -> impl nn::ModuleT {
    let mut dim = 3;
    let mut layers = nn::seq_t();

    for (i, cfg) in block_configs.iter().enumerate() {
        let q = p / "layers" / i;
        let is_last = i == block_configs.len() - 1;
        let dim_next = cfg.emb_dim;

        let block = nn::seq_t()
            .add(patch_embedding(&q / 0, dim, dim_next, cfg.patch_size))
            .add(transformer(
                &q / 1,
                dim_next,
                1,
                8,
                64,
                4,
                cfg.local_patch_size,
                cfg.global_k,
                dropout,
                !is_last,
            ))
            .add(peg(&q / 2, dim_next, peg_kernel_size))
            .add(transformer(
                &q / 3,
                dim_next,
                cfg.depth,
                8,
                64,
                4,
                cfg.local_patch_size,
                cfg.global_k,
                dropout,
                !is_last,
            ));

        layers = layers.add(block);
        dim = dim_next;
    }

    layers = layers
        .add_fn(|xs| xs.adaptive_avg_pool2d([1, 1]).squeeze())
        .add(nn::linear(
            p / "layers" / 19,
            dim,
            num_classes,
            Default::default(),
        ));

    nn::func_t(move |xs, train| xs.apply_t(&layers, train))
}
