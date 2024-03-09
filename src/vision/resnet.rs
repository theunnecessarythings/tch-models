/* Ported from TorchVision ResNet model.
 * ResNet model architecture from
 * `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_.
 */

use tch::nn::{self, ConvConfig};

fn conv3x3(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    stride: i64,
    groups: i64,
    dilation: i64,
) -> impl nn::Module {
    nn::conv2d(
        p,
        c_in,
        c_out,
        3,
        ConvConfig {
            stride,
            padding: dilation,
            groups,
            bias: false,
            dilation,
            ws_init: nn::Init::Kaiming {
                dist: nn::init::NormalOrUniform::Normal,
                fan: nn::init::FanInOut::FanOut,
                non_linearity: nn::init::NonLinearity::ReLU,
            },
            ..Default::default()
        },
    )
}

fn conv1x1(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl nn::Module {
    nn::conv2d(
        p,
        c_in,
        c_out,
        1,
        ConvConfig {
            stride,
            padding: 0,
            bias: false,
            ws_init: nn::Init::Kaiming {
                dist: nn::init::NormalOrUniform::Normal,
                fan: nn::init::FanInOut::FanOut,
                non_linearity: nn::init::NonLinearity::ReLU,
            },
            ..Default::default()
        },
    )
}

fn basic_block(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl nn::ModuleT {
    let conv1 = conv3x3(&p / "conv1", c_in, c_out, stride, 1, 1);
    let bn1 = nn::batch_norm2d(
        &p / "bn1",
        c_out,
        nn::BatchNormConfig {
            ws_init: nn::Init::Const(1.),
            ..Default::default()
        },
    );
    let conv2 = conv3x3(&p / "conv2", c_out, c_out, 1, 1, 1);
    let bn2 = nn::batch_norm2d(
        &p / "bn2",
        c_out,
        nn::BatchNormConfig {
            ws_init: nn::Init::Const(1.),
            ..Default::default()
        },
    );

    let downsample = if stride != 1 || (c_in != c_out) {
        nn::seq_t()
            .add(conv1x1(&p / "downsample" / 0, c_in, c_out, stride))
            .add(nn::batch_norm2d(
                &p / "downsample" / 1,
                c_out,
                nn::BatchNormConfig {
                    ws_init: nn::Init::Const(1.),
                    ..Default::default()
                },
            ))
    } else {
        nn::seq_t()
    };

    nn::func_t(move |xs, train| {
        let residual = xs.shallow_clone();
        let xs = xs
            .apply(&conv1)
            .apply_t(&bn1, train)
            .relu()
            .apply(&conv2)
            .apply_t(&bn2, train);
        let residual = residual.apply_t(&downsample, train);
        (xs + residual).relu()
    })
}

fn bottleneck(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    stride: i64,
    groups: i64,
    base_width: i64,
) -> impl nn::ModuleT {
    let width = (c_out as f64 * (base_width as f64 / 64.)).round() as i64 * groups;
    let conv1 = conv1x1(&p / "conv1", c_in, width, 1);
    let bn1 = nn::batch_norm2d(
        &p / "bn1",
        width,
        nn::BatchNormConfig {
            ws_init: nn::Init::Const(1.),
            ..Default::default()
        },
    );
    let conv2 = conv3x3(&p / "conv2", width, width, stride, groups, 1);
    let bn2 = nn::batch_norm2d(
        &p / "bn2",
        width,
        nn::BatchNormConfig {
            ws_init: nn::Init::Const(1.),
            ..Default::default()
        },
    );
    let conv3 = conv1x1(&p / "conv3", width, c_out * 4, 1);
    let bn3 = nn::batch_norm2d(
        &p / "bn3",
        c_out * 4,
        nn::BatchNormConfig {
            ws_init: nn::Init::Const(1.),
            ..Default::default()
        },
    );

    let downsample = if stride != 1 || (c_in != c_out * 4) {
        nn::seq_t()
            .add(conv1x1(&p / "downsample" / 0, c_in, c_out * 4, stride))
            .add(nn::batch_norm2d(
                &p / "downsample" / 1,
                c_out * 4,
                nn::BatchNormConfig {
                    ws_init: nn::Init::Const(1.),
                    ..Default::default()
                },
            ))
    } else {
        nn::seq_t()
    };

    nn::func_t(move |xs, train| {
        let residual = xs.shallow_clone();
        let xs = xs
            .apply(&conv1)
            .apply_t(&bn1, train)
            .relu()
            .apply(&conv2)
            .apply_t(&bn2, train)
            .relu()
            .apply(&conv3)
            .apply_t(&bn3, train);
        let residual = residual.apply_t(&downsample, train);
        (xs + residual).relu()
    })
}

enum BlockType {
    Basic,
    Bottleneck,
}

fn resnet(
    p: &nn::Path,
    block: BlockType,
    layers: &[i64],
    num_classes: i64,
    groups: i64,
    base_width: i64,
) -> impl nn::ModuleT {
    let c_in = 64;
    let expansion = match block {
        BlockType::Basic => 1,
        BlockType::Bottleneck => 4,
    };
    let conv1 = nn::conv2d(
        p / "conv1",
        3,
        c_in,
        7,
        ConvConfig {
            stride: 2,
            padding: 3,
            bias: false,
            ws_init: nn::Init::Kaiming {
                dist: nn::init::NormalOrUniform::Normal,
                fan: nn::init::FanInOut::FanOut,
                non_linearity: nn::init::NonLinearity::ReLU,
            },
            ..Default::default()
        },
    );

    let bn1 = nn::batch_norm2d(
        p / "bn1",
        c_in,
        nn::BatchNormConfig {
            ws_init: nn::Init::Const(1.),
            ..Default::default()
        },
    );
    let layer1 = make_layer(
        p / "layer1",
        &block,
        64,
        layers[0],
        1,
        groups,
        base_width,
        c_in,
    );
    let layer2 = make_layer(
        p / "layer2",
        &block,
        128,
        layers[1],
        2,
        groups,
        base_width,
        64 * expansion,
    );
    let layer3 = make_layer(
        p / "layer3",
        &block,
        256,
        layers[2],
        2,
        groups,
        base_width,
        128 * expansion,
    );
    let layer4 = make_layer(
        p / "layer4",
        &block,
        512,
        layers[3],
        2,
        groups,
        base_width,
        256 * expansion,
    );
    let fc = nn::linear(p / "fc", 512 * expansion, num_classes, Default::default());

    nn::func_t(move |xs, train| {
        xs.apply(&conv1)
            .apply_t(&bn1, train)
            .relu()
            .max_pool2d(3, 2, 1, 1, false)
            .apply_t(&layer1, train)
            .apply_t(&layer2, train)
            .apply_t(&layer3, train)
            .apply_t(&layer4, train)
            .adaptive_avg_pool2d([1, 1])
            .flat_view()
            .apply(&fc)
    })
}

fn make_layer(
    p: nn::Path,
    block: &BlockType,
    planes: i64,
    blocks: i64,
    stride: i64,
    groups: i64,
    base_width: i64,
    inplanes: i64,
) -> impl nn::ModuleT {
    let mut layers = nn::seq_t();
    match block {
        BlockType::Basic => {
            let block = basic_block(&p / 0, inplanes, planes, stride);
            layers = layers.add(block);
            let inplanes = planes;
            for _ in 1..blocks {
                let block = basic_block(&p / layers.len() as i64, inplanes, planes, 1);
                layers = layers.add(block);
            }
        }
        BlockType::Bottleneck => {
            let block = bottleneck(&p / 0, inplanes, planes, stride, groups, base_width);
            let inplanes = planes * 4;
            layers = layers.add(block);
            for _ in 1..blocks {
                let block = bottleneck(
                    &p / layers.len() as i64,
                    inplanes,
                    planes,
                    1,
                    groups,
                    base_width,
                );
                layers = layers.add(block);
            }
        }
    }
    layers
}

pub fn resnet18(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    resnet(p, BlockType::Basic, &[2, 2, 2, 2], num_classes, 1, 64)
}

pub fn resnet34(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    resnet(p, BlockType::Basic, &[3, 4, 6, 3], num_classes, 1, 64)
}

pub fn resnet50(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    resnet(p, BlockType::Bottleneck, &[3, 4, 6, 3], num_classes, 1, 64)
}

pub fn resnet101(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    resnet(p, BlockType::Bottleneck, &[3, 4, 23, 3], num_classes, 1, 64)
}

pub fn resnet152(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    resnet(p, BlockType::Bottleneck, &[3, 8, 36, 3], num_classes, 1, 64)
}

pub fn resnext50_32x4d(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    resnet(p, BlockType::Bottleneck, &[3, 4, 6, 3], num_classes, 32, 4)
}

pub fn resnext101_32x8d(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    resnet(p, BlockType::Bottleneck, &[3, 4, 23, 3], num_classes, 32, 8)
}

pub fn resnext101_64x4d(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    resnet(p, BlockType::Bottleneck, &[3, 4, 23, 3], num_classes, 64, 4)
}

pub fn wide_resnet50_2(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    resnet(
        p,
        BlockType::Bottleneck,
        &[3, 4, 6, 3],
        num_classes,
        1,
        64 * 2,
    )
}

pub fn wide_resnet101_2(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    resnet(
        p,
        BlockType::Bottleneck,
        &[3, 4, 23, 3],
        num_classes,
        1,
        64 * 2,
    )
}
