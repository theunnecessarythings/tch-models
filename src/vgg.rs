use tch::nn::{self, ConvConfig};

fn vgg(
    p: &nn::Path,
    num_classes: i64,
    cfg: Vec<Vec<i64>>,
    batch_norm: bool,
    dropout: f64,
) -> impl nn::ModuleT {
    let mut features = nn::seq_t();
    let mut c_in = 3;
    let mut idx = 0;
    let q = p / "features";
    for block in cfg {
        for c_out in block {
            features = features.add(nn::conv2d(
                &q / idx,
                c_in,
                c_out,
                3,
                ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ));
            idx += 1;
            if batch_norm {
                features = features.add(nn::batch_norm2d(&q / idx, c_out, Default::default()));
                idx += 1;
            }
            features = features.add_fn(|xs| xs.relu());
            idx += 1;
            c_in = c_out;
        }
        features = features.add_fn(|xs| xs.max_pool2d(2, 2, 0, 1, false));
        idx += 1;
    }

    let q = p / "classifier";
    let classifier = nn::seq_t()
        .add(nn::linear(&q / 0, 512 * 7 * 7, 4096, Default::default()))
        .add_fn_t(move |xs, train| xs.relu().dropout(dropout, train))
        .add(nn::linear(&q / 3, 4096, 4096, Default::default()))
        .add_fn_t(move |xs, train| xs.relu().dropout(dropout, train))
        .add(nn::linear(&q / 6, 4096, num_classes, Default::default()));
    nn::func_t(move |xs, train| {
        xs.apply_t(&features, train)
            .adaptive_avg_pool2d([7, 7])
            .flat_view()
            .apply_t(&classifier, train)
    })
}

enum VGGVersion {
    VGG11,
    VGG13,
    VGG16,
    VGG19,
}

fn get_vgg_cfg(kind: VGGVersion) -> Vec<Vec<i64>> {
    match kind {
        VGGVersion::VGG11 => vec![
            vec![64],
            vec![128],
            vec![256, 256],
            vec![512, 512],
            vec![512, 512],
        ],
        VGGVersion::VGG13 => vec![
            vec![64, 64],
            vec![128, 128],
            vec![256, 256],
            vec![512, 512],
            vec![512, 512],
        ],
        VGGVersion::VGG16 => vec![
            vec![64, 64],
            vec![128, 128],
            vec![256, 256, 256],
            vec![512, 512, 512],
            vec![512, 512, 512],
        ],
        VGGVersion::VGG19 => vec![
            vec![64, 64],
            vec![128, 128],
            vec![256, 256, 256, 256],
            vec![512, 512, 512, 512],
            vec![512, 512, 512, 512],
        ],
    }
}

pub fn vgg11(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    vgg(p, num_classes, get_vgg_cfg(VGGVersion::VGG11), false, 0.5)
}

pub fn vgg11_bn(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    vgg(p, num_classes, get_vgg_cfg(VGGVersion::VGG11), true, 0.5)
}

pub fn vgg13(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    vgg(p, num_classes, get_vgg_cfg(VGGVersion::VGG13), false, 0.5)
}

pub fn vgg13_bn(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    vgg(p, num_classes, get_vgg_cfg(VGGVersion::VGG13), true, 0.5)
}

pub fn vgg16(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    vgg(p, num_classes, get_vgg_cfg(VGGVersion::VGG16), false, 0.5)
}

pub fn vgg16_bn(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    vgg(p, num_classes, get_vgg_cfg(VGGVersion::VGG16), true, 0.5)
}

pub fn vgg19(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    vgg(p, num_classes, get_vgg_cfg(VGGVersion::VGG19), false, 0.5)
}

pub fn vgg19_bn(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    vgg(p, num_classes, get_vgg_cfg(VGGVersion::VGG19), true, 0.5)
}
