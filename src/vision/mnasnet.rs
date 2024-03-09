/* Ported from TorchVision MnasNet model.
 * Adapted from """MnasNet: Platform-Aware Neural Architecture Search for Mobile"""
 * <https://arxiv.org/abs/1807.11626>`_.
 */

use tch::{
    nn::{self, batch_norm2d, conv2d, init::DEFAULT_KAIMING_NORMAL, BatchNormConfig, ConvConfig},
    Kind,
};

fn inverted_residual(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    ksize: i64,
    stride: i64,
    expansion_factor: i64,
    bn_momentum: f64,
) -> impl nn::ModuleT {
    let mid_ch = c_in * expansion_factor;
    let apply_residual = c_in == c_out && stride == 1;
    let p = p / "layers";
    let layers = nn::seq_t()
        .add(conv2d(
            &p / 0,
            c_in,
            mid_ch,
            1,
            ConvConfig {
                bias: false,
                ..Default::default()
            },
        ))
        .add(batch_norm2d(
            &p / 1,
            mid_ch,
            nn::BatchNormConfig {
                momentum: bn_momentum,
                ..Default::default()
            },
        ))
        .add_fn(|xs| xs.relu())
        .add(conv2d(
            &p / 3,
            mid_ch,
            mid_ch,
            ksize,
            ConvConfig {
                groups: mid_ch,
                stride,
                padding: ksize / 2,
                bias: false,
                ..Default::default()
            },
        ))
        .add(batch_norm2d(
            &p / 4,
            mid_ch,
            nn::BatchNormConfig {
                momentum: bn_momentum,
                ..Default::default()
            },
        ))
        .add_fn(|xs| xs.relu())
        .add(conv2d(
            &p / 6,
            mid_ch,
            c_out,
            1,
            ConvConfig {
                bias: false,
                ..Default::default()
            },
        ))
        .add(batch_norm2d(
            &p / 7,
            c_out,
            nn::BatchNormConfig {
                momentum: bn_momentum,
                ..Default::default()
            },
        ));
    nn::func_t(move |xs, train| {
        let ys = xs.apply_t(&layers, train);
        if apply_residual {
            xs + ys
        } else {
            ys
        }
    })
}

fn stack(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    ksize: i64,
    stride: i64,
    exp_factor: i64,
    repeats: i64,
    bn_momentum: f64,
) -> impl nn::ModuleT {
    let mut layers = nn::seq_t();
    layers = layers.add(inverted_residual(
        &p / 0,
        c_in,
        c_out,
        ksize,
        stride,
        exp_factor,
        bn_momentum,
    ));
    for i in 1..repeats {
        layers = layers.add(inverted_residual(
            &p / i,
            c_out,
            c_out,
            ksize,
            1,
            exp_factor,
            bn_momentum,
        ));
    }
    layers
}

fn round_to_multiple_of(val: f64, divisor: i64) -> i64 {
    let new_val = i64::max(
        divisor,
        (val + divisor as f64 / 2.0) as i64 / divisor * divisor,
    );
    if new_val as f64 >= 0.9 * val {
        new_val
    } else {
        new_val + divisor
    }
}

fn get_depths(alpha: f64) -> Vec<i64> {
    [32, 16, 24, 40, 80, 96, 192, 320]
        .iter()
        .map(|&x| round_to_multiple_of(x as f64 * alpha, 8))
        .collect()
}

fn mnasnet(p: &nn::Path, alpha: f64, num_classes: i64, dropout: f64) -> impl nn::ModuleT {
    let bn_momentum = 1. - 0.9997;
    let depths = get_depths(alpha);
    let q = p / "layers";
    let layers = nn::seq_t()
        .add(conv2d(
            &q / 0,
            3,
            depths[0],
            3,
            ConvConfig {
                stride: 2,
                padding: 1,
                bias: false,
                ws_init: DEFAULT_KAIMING_NORMAL,
                ..Default::default()
            },
        ))
        .add(batch_norm2d(
            &q / 1,
            depths[0],
            BatchNormConfig {
                momentum: bn_momentum,
                ws_init: nn::Init::Const(1.0),
                ..Default::default()
            },
        ))
        .add_fn(|xs| xs.relu())
        .add(conv2d(
            &q / 3,
            depths[0],
            depths[0],
            3,
            ConvConfig {
                stride: 1,
                padding: 1,
                groups: depths[0],
                bias: false,
                ws_init: DEFAULT_KAIMING_NORMAL,
                ..Default::default()
            },
        ))
        .add(batch_norm2d(
            &q / 4,
            depths[0],
            BatchNormConfig {
                momentum: bn_momentum,
                ws_init: nn::Init::Const(1.0),
                ..Default::default()
            },
        ))
        .add_fn(|xs| xs.relu())
        .add(conv2d(
            &q / 6,
            depths[0],
            depths[1],
            1,
            ConvConfig {
                stride: 1,
                padding: 0,
                bias: false,
                ws_init: DEFAULT_KAIMING_NORMAL,
                ..Default::default()
            },
        ))
        .add(batch_norm2d(
            &q / 7,
            depths[1],
            BatchNormConfig {
                momentum: bn_momentum,
                ws_init: nn::Init::Const(1.0),
                ..Default::default()
            },
        ))
        .add(stack(&q / 8, depths[1], depths[2], 3, 2, 3, 3, bn_momentum))
        .add(stack(&q / 9, depths[2], depths[3], 5, 2, 3, 3, bn_momentum))
        .add(stack(
            &q / 10,
            depths[3],
            depths[4],
            5,
            2,
            6,
            3,
            bn_momentum,
        ))
        .add(stack(
            &q / 11,
            depths[4],
            depths[5],
            3,
            1,
            6,
            2,
            bn_momentum,
        ))
        .add(stack(
            &q / 12,
            depths[5],
            depths[6],
            5,
            2,
            6,
            4,
            bn_momentum,
        ))
        .add(stack(
            &q / 13,
            depths[6],
            depths[7],
            3,
            1,
            6,
            1,
            bn_momentum,
        ))
        .add(conv2d(
            &q / 14,
            depths[7],
            1280,
            1,
            ConvConfig {
                stride: 1,
                padding: 0,
                bias: false,
                ws_init: DEFAULT_KAIMING_NORMAL,
                ..Default::default()
            },
        ))
        .add(batch_norm2d(
            &q / 15,
            1280,
            BatchNormConfig {
                momentum: bn_momentum,
                ws_init: nn::Init::Const(1.0),
                ..Default::default()
            },
        ))
        .add_fn(|xs| xs.relu());
    let classifier = nn::seq_t()
        .add_fn_t(move |xs, train| xs.dropout(dropout, train))
        .add(nn::linear(
            p / "classifier" / 1,
            1280,
            num_classes,
            nn::LinearConfig {
                ws_init: nn::Init::Kaiming {
                    dist: nn::init::NormalOrUniform::Uniform,
                    fan: nn::init::FanInOut::FanOut,
                    non_linearity: nn::init::NonLinearity::Sigmoid,
                },
                ..Default::default()
            },
        ));
    nn::func_t(move |xs, train| {
        let ys = xs.apply_t(&layers, train);
        let ys = ys.mean_dim(vec![2, 3], false, Kind::Float);
        ys.apply_t(&classifier, train)
    })
}

pub fn mnasnet0_5(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    mnasnet(p, 0.5, num_classes, 0.2)
}

pub fn mnasnet0_75(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    mnasnet(p, 0.75, num_classes, 0.2)
}

pub fn mnasnet1_0(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    mnasnet(p, 1.0, num_classes, 0.2)
}

pub fn mnasnet1_3(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    mnasnet(p, 1.3, num_classes, 0.2)
}
