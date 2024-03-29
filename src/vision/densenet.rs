//! DenseNet implementation ported from torchvision library.
//!
//! See "Densely Connected Convolutional Networks", Huang et al 2016.
//! <https://arxiv.org/abs/1608.06993>
//!
use tch::{
    nn::{self, init::DEFAULT_KAIMING_NORMAL, Conv2D, ModuleT},
    Tensor,
};

fn conv2d(p: nn::Path, c_in: i64, c_out: i64, ksize: i64, padding: i64, stride: i64) -> Conv2D {
    let conv2d_cfg = nn::ConvConfig {
        stride,
        padding,
        bias: false,
        ws_init: DEFAULT_KAIMING_NORMAL,
        ..Default::default()
    };
    nn::conv2d(p, c_in, c_out, ksize, conv2d_cfg)
}

fn dense_layer(p: nn::Path, c_in: i64, bn_size: i64, growth: i64) -> impl ModuleT {
    let c_inter = bn_size * growth;
    let bn1 = nn::batch_norm2d(
        &p / "norm1",
        c_in,
        nn::BatchNormConfig {
            ws_init: nn::Init::Const(1.0),
            ..Default::default()
        },
    );
    let conv1 = conv2d(&p / "conv1", c_in, c_inter, 1, 0, 1);
    let bn2 = nn::batch_norm2d(
        &p / "norm2",
        c_inter,
        nn::BatchNormConfig {
            ws_init: nn::Init::Const(1.0),
            ..Default::default()
        },
    );
    let conv2 = conv2d(&p / "conv2", c_inter, growth, 3, 1, 1);
    nn::func_t(move |xs, train| {
        xs.apply_t(&bn1, train)
            .relu()
            .apply(&conv1)
            .apply_t(&bn2, train)
            .relu()
            .apply(&conv2)
    })
}

fn dense_block(p: nn::Path, c_in: i64, bn_size: i64, growth: i64, nlayers: i64) -> impl ModuleT {
    let mut layers = vec![];
    for i in 0..nlayers {
        layers.push(dense_layer(
            &p / &format!("denselayer{}", 1 + i),
            c_in + i * growth,
            bn_size,
            growth,
        ));
    }
    nn::func_t(move |xs, train| {
        let mut features = vec![xs * 1];
        for layer in &layers {
            let ys = layer.forward_t(&Tensor::cat(&features, 1), train);
            features.push(ys);
        }
        Tensor::cat(&features, 1)
    })
}

fn transition(p: nn::Path, c_in: i64, c_out: i64) -> impl ModuleT {
    nn::seq_t()
        .add(nn::batch_norm2d(
            &p / "norm",
            c_in,
            nn::BatchNormConfig {
                ws_init: nn::Init::Const(1.0),
                ..Default::default()
            },
        ))
        .add_fn(|xs| xs.relu())
        .add(conv2d(&p / "conv", c_in, c_out, 1, 0, 1))
        .add_fn(|xs| xs.avg_pool2d(2, 2, 0, false, true, None))
}

fn densenet(
    p: &nn::Path,
    c_in: i64,
    bn_size: i64,
    growth: i64,
    block_config: &[i64],
    c_out: i64,
) -> impl ModuleT {
    let fp = p / "features";
    let mut seq = nn::seq_t()
        .add(conv2d(&fp / "conv0", 3, c_in, 7, 3, 2))
        .add(nn::batch_norm2d(
            &fp / "norm0",
            c_in,
            nn::BatchNormConfig {
                ws_init: nn::Init::Const(1.0),
                ..Default::default()
            },
        ))
        .add_fn(|xs| xs.relu().max_pool2d([3, 3], [2, 2], [1, 1], [1, 1], false));
    let mut nfeat = c_in;

    for (i, &nlayers) in block_config.iter().enumerate() {
        seq = seq.add(dense_block(
            &fp / &format!("denseblock{}", 1 + i),
            nfeat,
            bn_size,
            growth,
            nlayers,
        ));
        nfeat += nlayers * growth;
        if i + 1 != block_config.len() {
            seq = seq.add(transition(
                &fp / &format!("transition{}", 1 + i),
                nfeat,
                nfeat / 2,
            ));
            nfeat /= 2
        }
    }
    seq.add(nn::batch_norm2d(
        &fp / "norm5",
        nfeat,
        nn::BatchNormConfig {
            ws_init: nn::Init::Const(1.0),
            ..Default::default()
        },
    ))
    .add_fn(|xs| xs.relu().adaptive_avg_pool2d([1, 1]).flat_view())
    .add(nn::linear(
        p / "classifier",
        nfeat,
        c_out,
        nn::LinearConfig {
            bs_init: Some(nn::Init::Const(0.0)),
            ..Default::default()
        },
    ))
}

pub fn densenet121(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    densenet(p, 64, 4, 32, &[6, 12, 24, 16], nclasses)
}

pub fn densenet161(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    densenet(p, 96, 4, 48, &[6, 12, 36, 24], nclasses)
}

pub fn densenet169(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    densenet(p, 64, 4, 32, &[6, 12, 32, 32], nclasses)
}

pub fn densenet201(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    densenet(p, 64, 4, 32, &[6, 12, 48, 32], nclasses)
}
