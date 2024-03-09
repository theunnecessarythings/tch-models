/* Ported from torchvision
* SqueezeNet model architecture from
* `SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size <https://arxiv.org/abs/1602.07360>`_.
*/

use tch::{
    nn::{self, ConvConfig},
    Tensor,
};

fn fire(
    p: nn::Path,
    c_in: i64,
    c_squeeze: i64,
    c_expand1x1: i64,
    c_expand3x3: i64,
) -> impl nn::ModuleT {
    let squeeze = nn::conv2d(&p / "squeeze", c_in, c_squeeze, 1, Default::default());
    let expand1x1 = nn::conv2d(
        &p / "expand1x1",
        c_squeeze,
        c_expand1x1,
        1,
        Default::default(),
    );
    let expand3x3 = nn::conv2d(
        &p / "expand3x3",
        c_squeeze,
        c_expand3x3,
        3,
        ConvConfig {
            padding: 1,
            ..Default::default()
        },
    );
    nn::func_t(move |xs, _| {
        let xs = xs.apply(&squeeze).relu();
        Tensor::cat(
            &[xs.apply(&expand1x1).relu(), xs.apply(&expand3x3).relu()],
            1,
        )
    })
}

enum SqueezeNetVersion {
    SqueezeNet1_0,
    SqueezeNet1_1,
}

fn squeeze_net(
    p: &nn::Path,
    version: SqueezeNetVersion,
    num_classes: i64,
    dropout: f64,
) -> impl nn::ModuleT {
    let q = p / "features";
    let features = match version {
        SqueezeNetVersion::SqueezeNet1_0 => nn::seq_t()
            .add(nn::conv2d(
                &q / 0,
                3,
                96,
                7,
                ConvConfig {
                    stride: 2,
                    ..Default::default()
                },
            ))
            .add_fn(|xs| xs.relu().max_pool2d(3, 2, 0, 1, false))
            .add(fire(&q / 3, 96, 16, 64, 64))
            .add(fire(&q / 4, 128, 16, 64, 64))
            .add(fire(&q / 5, 128, 32, 128, 128))
            .add_fn(|xs| xs.max_pool2d(3, 2, 0, 1, false))
            .add(fire(&q / 7, 256, 32, 128, 128))
            .add(fire(&q / 8, 256, 48, 192, 192))
            .add(fire(&q / 9, 384, 48, 192, 192))
            .add(fire(&q / 10, 384, 64, 256, 256))
            .add_fn(|xs| xs.max_pool2d(3, 2, 0, 1, false))
            .add(fire(&q / 12, 512, 64, 256, 256)),
        SqueezeNetVersion::SqueezeNet1_1 => nn::seq_t()
            .add(nn::conv2d(
                &q / 0,
                3,
                64,
                3,
                ConvConfig {
                    stride: 2,
                    ..Default::default()
                },
            ))
            .add_fn(|xs| xs.relu().max_pool2d(3, 2, 0, 1, false))
            .add(fire(&q / 3, 64, 16, 64, 64))
            .add(fire(&q / 4, 128, 16, 64, 64))
            .add_fn(|xs| xs.max_pool2d(3, 2, 0, 1, false))
            .add(fire(&q / 6, 128, 32, 128, 128))
            .add(fire(&q / 7, 256, 32, 128, 128))
            .add_fn(|xs| xs.max_pool2d(3, 2, 0, 1, false))
            .add(fire(&q / 9, 256, 48, 192, 192))
            .add(fire(&q / 10, 384, 48, 192, 192))
            .add(fire(&q / 11, 384, 64, 256, 256))
            .add(fire(&q / 12, 512, 64, 256, 256)),
    };
    let final_conv = nn::conv2d(
        p / "classifier" / 1,
        512,
        num_classes,
        1,
        ConvConfig {
            ws_init: nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
            ..Default::default()
        },
    );
    nn::seq_t().add_fn_t(move |xs, train| {
        xs.apply_t(&features, train)
            .dropout(dropout, train)
            .apply(&final_conv)
            .relu()
            .adaptive_avg_pool2d([1, 1])
            .flat_view()
    })
}

pub fn squeezenet1_0(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    squeeze_net(p, SqueezeNetVersion::SqueezeNet1_0, num_classes, 0.5)
}

pub fn squeezenet1_1(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    squeeze_net(p, SqueezeNetVersion::SqueezeNet1_1, num_classes, 0.5)
}
