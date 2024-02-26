use tch::nn::{self, batch_norm2d, conv2d, BatchNormConfig, ConvConfig};

fn make_divisible(v: f64, divisor: i64, min_value: Option<i64>) -> i64 {
    let min_value = min_value.unwrap_or(divisor);
    let new_v = i64::max(
        min_value,
        (v + divisor as f64 / 2.0) as i64 / divisor * divisor,
    );
    if (new_v as f64) < (0.9 * v) {
        new_v + divisor
    } else {
        new_v
    }
}

#[derive(Debug, Clone)]
enum Activation {
    Hardswish,
    Relu,
    None,
}

fn conv_norm_activation(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    ksize: i64,
    stride: i64,
    groups: i64,
    dilation: i64,
    activation: Activation,
) -> impl nn::ModuleT {
    let padding = (ksize - 1) / 2 * dilation;
    nn::seq_t()
        .add(conv2d(
            &p / 0,
            c_in,
            c_out,
            ksize,
            ConvConfig {
                stride,
                groups,
                padding,
                bias: false,
                dilation,
                ..Default::default()
            },
        ))
        .add(batch_norm2d(&p / 1, c_out, BatchNormConfig::default()))
        .add_fn(move |xs| match activation {
            Activation::Hardswish => xs.hardswish(),
            Activation::Relu => xs.relu(),
            Activation::None => xs.shallow_clone(),
        })
}

#[derive(Debug, Clone)]
struct InvertedResidualConfig {
    c_in: i64,
    ksize: i64,
    expanded_channels: i64,
    c_out: i64,
    use_se: bool,
    use_hs: bool,
    stride: i64,
    dilation: i64,
    width_mult: f64,
}

impl InvertedResidualConfig {
    fn adjust_channels(c: i64, width_mult: f64) -> i64 {
        make_divisible(c as f64 * width_mult, 8, None)
    }

    fn new(
        c_in: i64,
        ksize: i64,
        expanded_channels: i64,
        c_out: i64,
        use_se: bool,
        use_hs: bool,
        stride: i64,
        dilation: i64,
        width_mult: f64,
    ) -> Self {
        let c_in = Self::adjust_channels(c_in, width_mult);
        let expanded_channels = Self::adjust_channels(expanded_channels, width_mult);
        let c_out = Self::adjust_channels(c_out, width_mult);
        InvertedResidualConfig {
            c_in,
            ksize,
            expanded_channels,
            c_out,
            use_se,
            use_hs,
            stride,
            dilation,
            width_mult,
        }
    }
}

fn squeeze_excitation(p: nn::Path, c_in: i64, c_squeeze: i64) -> impl nn::ModuleT {
    let scale = nn::seq_t()
        .add_fn(|xs| xs.adaptive_avg_pool2d([1, 1]))
        .add(conv2d(&p / "fc1", c_in, c_squeeze, 1, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(conv2d(&p / "fc2", c_squeeze, c_in, 1, Default::default()))
        .add_fn(|xs| xs.hardsigmoid());
    nn::func_t(move |xs, train| xs * xs.apply_t(&scale, train))
}

fn inverted_residual(p: nn::Path, cfg: &InvertedResidualConfig) -> impl nn::ModuleT {
    let use_res_connect = cfg.stride == 1 && cfg.c_in == cfg.c_out;
    let mut layers = nn::seq_t();
    let p = p / "block";
    let mut idx = 0;
    let activation = if cfg.use_hs {
        Activation::Hardswish
    } else {
        Activation::Relu
    };
    if cfg.expanded_channels != cfg.c_in {
        layers = layers.add(conv_norm_activation(
            &p / idx,
            cfg.c_in,
            cfg.expanded_channels,
            1,
            1,
            1,
            1,
            activation.clone(),
        ));
        idx += 1;
    }
    let stride = if cfg.dilation > 1 { 1 } else { cfg.stride };
    layers = layers.add(conv_norm_activation(
        &p / idx,
        cfg.expanded_channels,
        cfg.expanded_channels,
        cfg.ksize,
        stride,
        cfg.expanded_channels,
        cfg.dilation,
        activation.clone(),
    ));
    idx += 1;

    if cfg.use_se {
        let c_squeeze = make_divisible((cfg.expanded_channels / 4) as f64, 8, None);
        layers = layers.add(squeeze_excitation(
            &p / idx,
            cfg.expanded_channels,
            c_squeeze,
        ));
        idx += 1;
    }

    layers = layers.add(conv_norm_activation(
        &p / idx,
        cfg.expanded_channels,
        cfg.c_out,
        1,
        1,
        1,
        1,
        Activation::None,
    ));

    nn::func_t(move |xs, train| {
        let result = xs.apply_t(&layers, train);
        if use_res_connect {
            xs + result
        } else {
            result
        }
    })
}

fn mobilenet_v3(
    p: &nn::Path,
    inverted_residual_setting: &[InvertedResidualConfig],
    last_channel: i64,
    num_classes: i64,
    dropout: f64,
) -> impl nn::ModuleT {
    let mut layers = nn::seq_t();
    let firstconv_output_channels = inverted_residual_setting[0].c_in;
    let q = p / "features";
    layers = layers.add(conv_norm_activation(
        &q / 0,
        3,
        firstconv_output_channels,
        3,
        2,
        1,
        1,
        Activation::Hardswish,
    ));

    for (i, cfg) in inverted_residual_setting.iter().enumerate() {
        layers = layers.add(inverted_residual(&q / (i + 1), cfg));
    }
    let lastconv_input_channels = inverted_residual_setting.last().unwrap().c_out;
    let lastconv_output_channels = 6 * lastconv_input_channels;
    let idx = layers.len();
    layers = layers.add(conv_norm_activation(
        q / idx,
        lastconv_input_channels,
        lastconv_output_channels,
        1,
        1,
        1,
        1,
        Activation::Hardswish,
    ));
    let classifier = nn::seq_t()
        .add(nn::linear(
            p / "classifier" / 0,
            lastconv_output_channels,
            last_channel,
            Default::default(),
        ))
        .add_fn(|xs| xs.hardswish())
        .add_fn_t(move |xs, train| xs.dropout(dropout, train))
        .add(nn::linear(
            p / "classifier" / 3,
            last_channel,
            num_classes,
            Default::default(),
        ));
    nn::func_t(move |xs, train| {
        xs.apply_t(&layers, train)
            .adaptive_avg_pool2d([1, 1])
            .flat_view()
            .apply_t(&classifier, train)
    })
}

enum MobilenetV3Kind {
    Small,
    Large,
}

fn mobilenet_v3_conf(
    arch: MobilenetV3Kind,
    width_mult: f64,
    reduced_tail: bool,
    dilated: bool,
) -> (Vec<InvertedResidualConfig>, i64) {
    let reduce_divider = if reduced_tail { 2 } else { 1 };
    let dilation = if dilated { 2 } else { 1 };
    let inverted_residual_setting = match arch {
        MobilenetV3Kind::Large => vec![
            InvertedResidualConfig::new(16, 3, 16, 16, false, false, 1, 1, width_mult),
            InvertedResidualConfig::new(16, 3, 64, 24, false, false, 2, 1, width_mult),
            InvertedResidualConfig::new(24, 3, 72, 24, false, false, 1, 1, width_mult),
            InvertedResidualConfig::new(24, 5, 72, 40, true, false, 2, 1, width_mult),
            InvertedResidualConfig::new(40, 5, 120, 40, true, false, 1, 1, width_mult),
            InvertedResidualConfig::new(40, 5, 120, 40, true, false, 1, 1, width_mult),
            InvertedResidualConfig::new(40, 3, 240, 80, false, true, 2, 1, width_mult),
            InvertedResidualConfig::new(80, 3, 200, 80, false, true, 1, 1, width_mult),
            InvertedResidualConfig::new(80, 3, 184, 80, false, true, 1, 1, width_mult),
            InvertedResidualConfig::new(80, 3, 184, 80, false, true, 1, 1, width_mult),
            InvertedResidualConfig::new(80, 3, 480, 112, true, true, 1, 1, width_mult),
            InvertedResidualConfig::new(112, 3, 672, 112, true, true, 1, 1, width_mult),
            InvertedResidualConfig::new(
                112,
                5,
                672,
                160 / reduce_divider,
                true,
                true,
                2,
                dilation,
                width_mult,
            ),
            InvertedResidualConfig::new(
                160 / reduce_divider,
                5,
                960 / reduce_divider,
                160 / reduce_divider,
                true,
                true,
                1,
                dilation,
                width_mult,
            ),
            InvertedResidualConfig::new(
                160 / reduce_divider,
                5,
                960 / reduce_divider,
                160 / reduce_divider,
                true,
                true,
                1,
                dilation,
                width_mult,
            ),
        ],
        MobilenetV3Kind::Small => vec![
            InvertedResidualConfig::new(16, 3, 16, 16, true, false, 2, 1, width_mult),
            InvertedResidualConfig::new(16, 3, 72, 24, false, false, 2, 1, width_mult),
            InvertedResidualConfig::new(24, 3, 88, 24, false, false, 1, 1, width_mult),
            InvertedResidualConfig::new(24, 5, 96, 40, true, true, 2, 1, width_mult),
            InvertedResidualConfig::new(40, 5, 240, 40, true, true, 1, 1, width_mult),
            InvertedResidualConfig::new(40, 5, 240, 40, true, true, 1, 1, width_mult),
            InvertedResidualConfig::new(40, 5, 120, 48, true, true, 1, 1, width_mult),
            InvertedResidualConfig::new(48, 5, 144, 48, true, true, 1, 1, width_mult),
            InvertedResidualConfig::new(
                48,
                5,
                288,
                96 / reduce_divider,
                true,
                true,
                2,
                dilation,
                width_mult,
            ),
            InvertedResidualConfig::new(
                96 / reduce_divider,
                5,
                576 / reduce_divider,
                96 / reduce_divider,
                true,
                true,
                1,
                dilation,
                width_mult,
            ),
            InvertedResidualConfig::new(
                96 / reduce_divider,
                5,
                576 / reduce_divider,
                96 / reduce_divider,
                true,
                true,
                1,
                dilation,
                width_mult,
            ),
        ],
    };
    let last_channel = match arch {
        MobilenetV3Kind::Large => {
            InvertedResidualConfig::adjust_channels(1280 / reduce_divider, width_mult)
        }
        MobilenetV3Kind::Small => {
            InvertedResidualConfig::adjust_channels(1024 / reduce_divider, width_mult)
        }
    };
    (inverted_residual_setting, last_channel)
}

pub fn mobilenet_v3_small(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let (inverted_residual_setting, last_channel) =
        mobilenet_v3_conf(MobilenetV3Kind::Small, 1.0, false, false);
    mobilenet_v3(
        p,
        &inverted_residual_setting,
        last_channel,
        num_classes,
        0.2,
    )
}

pub fn mobilenet_v3_large(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    let (inverted_residual_setting, last_channel) =
        mobilenet_v3_conf(MobilenetV3Kind::Large, 1.0, false, false);
    mobilenet_v3(
        p,
        &inverted_residual_setting,
        last_channel,
        num_classes,
        0.2,
    )
}
