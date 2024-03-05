use tch::{nn, IndexOp, Kind, Tensor};

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
    let norm = nn::layer_norm(&p / "norm", vec![dim], Default::default());
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
        ys.apply(&norm)
    })
}

fn rearrange_image(x: &Tensor) -> Tensor {
    let (b, hw, c) = x.size3().unwrap();
    let h = (hw as f64).sqrt() as i64;
    x.view([b, h, hw / h, c]).permute([0, 3, 1, 2])
}

pub enum Pool {
    Cls,
    Mean,
}

fn conv_output_size(image_size: i64, ksize: i64, stride: i64, padding: i64) -> i64 {
    ((image_size - ksize + (2 * padding)) / stride) + 1
}

pub fn t2t_vit(
    p: &nn::Path,
    image_size: i64,
    num_classes: i64,
    dim: i64,
    depth: Option<i64>,
    heads: Option<i64>,
    mlp_dim: Option<i64>,
    pool: Pool,
    channels: i64,
    head_dim: i64,
    dropout: f64,
    emb_dropout: f64,
    transformer_block: Option<Box<dyn nn::ModuleT>>,
    t2t_layers: &'static [(i64, i64)],
) -> impl nn::ModuleT {
    let mut layers = nn::seq_t();
    let mut layer_dim = channels;
    let mut output_image_size = image_size;

    let q = p / "to_patch_embedding";
    for (i, (ksize, stride)) in t2t_layers.iter().enumerate() {
        layer_dim *= ksize * ksize;
        let is_first = i == 0;
        let is_last = i == t2t_layers.len() - 1;
        output_image_size = conv_output_size(output_image_size, *ksize, *stride, *stride / 2);

        layers = layers.add_fn(move |xs| {
            if !is_first {
                rearrange_image(xs)
            } else {
                xs.shallow_clone()
            }
            .im2col(
                [*ksize, *ksize],
                [1, 1],
                [stride / 2, stride / 2],
                [*stride, *stride],
            )
            .permute([0, 2, 1])
        });
        if !is_last {
            let block = transformer(
                &q / layers.len(),
                layer_dim,
                1,
                1,
                layer_dim,
                layer_dim,
                dropout,
            );
            layers = layers.add(block);
        }
    }

    let idx = layers.len();
    layers = layers.add(nn::linear(&q / idx, layer_dim, dim, Default::default()));

    let pos_embedding = p.var(
        "pos_embedding",
        &[1, output_image_size.pow(2) + 1, dim],
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

    let transformer = match transformer_block.is_none() {
        true => {
            assert!(
                depth.is_some() && heads.is_some() && mlp_dim.is_some(),
                "depth, heads, and mlp_dim must be provided if transformer_block is not provided"
            );
            Box::new(transformer(
                p / "transformer",
                dim,
                depth.unwrap(),
                heads.unwrap(),
                head_dim,
                mlp_dim.unwrap(),
                dropout,
            ))
        }
        false => transformer_block.unwrap(),
    };

    let q = p / "mlp_head";
    let mlp_head = nn::seq()
        .add(nn::layer_norm(&q / 0, vec![dim], Default::default()))
        .add(nn::linear(&q / 1, dim, num_classes, Default::default()));

    nn::func_t(move |xs, train| {
        let mut ys = xs.apply_t(&layers, train);
        let (b, n, _) = ys.size3().unwrap();
        let cls_tokens = cls_token.repeat([b, 1, 1]);
        ys = Tensor::cat(&[cls_tokens, ys], 1);
        ys += &pos_embedding.i((.., ..(n + 1)));
        ys = ys.dropout(emb_dropout, train);
        ys = transformer.forward_t(&ys, train);
        match pool {
            Pool::Cls => ys.i((.., 0)),
            Pool::Mean => ys.mean_dim(1, false, Kind::Float),
        }
        .apply(&mlp_head)
    })
}
