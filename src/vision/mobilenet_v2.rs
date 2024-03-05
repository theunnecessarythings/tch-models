use tch::nn::{self, batch_norm2d, conv2d, BatchNormConfig, ConvConfig};

fn make_divisible(v: f64, divisor: i64, min_value: Option<i64>) -> i64 {
    let min_value = min_value.unwrap_or(divisor);
    let new_v = i64::max(
        min_value,
        (v + divisor as f64 / 2.0) as i64 / divisor * divisor,
    );
    if new_v < (0.9 * v) as i64 {
        new_v + divisor
    } else {
        new_v
    }
}

fn conv_norm_activation(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    ksize: i64,
    stride: i64,
    groups: i64,
) -> impl nn::ModuleT {
    let padding = (ksize - 1) / 2;
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
                ..Default::default()
            },
        ))
        .add(batch_norm2d(&p / 1, c_out, BatchNormConfig::default()))
        .add_fn(|xs| xs.relu6())
}
fn inverted_residual(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    stride: i64,
    expand_ratio: i64,
) -> impl nn::ModuleT {
    let hidden_dim = c_in * expand_ratio;
    let use_res_connect = stride == 1 && c_in == c_out;
    let mut layers = nn::seq_t();
    let p = p / "conv";
    let mut idx = 0;
    if expand_ratio != 1 {
        layers = layers.add(conv_norm_activation(&p / idx, c_in, hidden_dim, 1, 1, 1));
        idx += 1;
    }
    layers = layers
        .add(conv_norm_activation(
            &p / idx,
            hidden_dim,
            hidden_dim,
            3,
            stride,
            hidden_dim,
        ))
        .add(nn::conv2d(
            &p / (idx + 1),
            hidden_dim,
            c_out,
            1,
            ConvConfig {
                bias: false,
                ..Default::default()
            },
        ))
        .add(batch_norm2d(&p / (idx + 2), c_out, Default::default()));

    nn::func_t(move |xs, train| {
        if use_res_connect {
            xs + xs.apply_t(&layers, train)
        } else {
            xs.apply_t(&layers, train)
        }
    })
}

fn mobilenet(
    p: &nn::Path,
    num_classes: i64,
    width_mult: f64,
    round_nearest: i64,
    dropout: f64,
) -> impl nn::ModuleT {
    let c_in = 32;
    let last_channel = 1280;
    let inverted_residual_setting = vec![
        // t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ];
    let mut c_in = make_divisible(c_in as f64 * width_mult, round_nearest, None);
    let last_channel = make_divisible(
        last_channel as f64 * f64::max(1.0, width_mult),
        round_nearest,
        None,
    );
    let mut layers = nn::seq_t();
    let q = p / "features";
    layers = layers.add(conv_norm_activation(&q / 0, 3, c_in, 3, 2, 1));
    for &[t, c, n, s] in &inverted_residual_setting {
        let c_out = make_divisible(c as f64 * width_mult, round_nearest, None);
        for i in 0..n {
            let stride = if i == 0 { s } else { 1 };
            let idx = layers.len();
            layers = layers.add(inverted_residual(&q / idx, c_in, c_out, stride, t));
            c_in = c_out;
        }
    }
    layers = layers.add(conv_norm_activation(&q / 18, c_in, last_channel, 1, 1, 1));
    let fc = nn::linear(
        p / "classifier" / 1,
        last_channel,
        num_classes,
        Default::default(),
    );
    nn::func_t(move |xs, train| {
        xs.apply_t(&layers, train)
            .adaptive_avg_pool2d([1, 1])
            .flat_view()
            .dropout(dropout, train)
            .apply(&fc)
    })
}

pub fn mobilenet_v2(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    mobilenet(p, num_classes, 1.0, 8, 0.2)
}
