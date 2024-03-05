use tch::{
    nn::{self, ConvConfig},
    Tensor,
};

fn channel_shuffle(x: &Tensor, groups: i64) -> Tensor {
    let (batchsize, num_channels, height, width) = x.size4().unwrap();
    let channels_per_group = num_channels / groups;
    let x = x.view([batchsize, groups, channels_per_group, height, width]);
    let x = x.transpose(1, 2).contiguous();
    x.view([batchsize, num_channels, height, width])
}

fn depthwise_conv(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    ksize: i64,
    stride: i64,
    padding: i64,
    bias: bool,
) -> impl nn::ModuleT {
    nn::conv2d(
        p,
        c_in,
        c_out,
        ksize,
        ConvConfig {
            stride,
            padding,
            groups: c_in,
            bias,
            ..Default::default()
        },
    )
}

fn inverted_residual(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl nn::ModuleT {
    let branch_features = c_out / 2;
    let q = &p / "branch1";
    let branch1 = if stride > 1 {
        nn::seq_t()
            .add(depthwise_conv(&q / 0, c_in, c_in, 3, stride, 1, false))
            .add(nn::batch_norm2d(&q / 1, c_in, Default::default()))
            .add(nn::conv2d(
                &q / 2,
                c_in,
                branch_features,
                1,
                ConvConfig {
                    bias: false,
                    ..Default::default()
                },
            ))
            .add(nn::batch_norm2d(
                &q / 3,
                branch_features,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
    } else {
        nn::seq_t()
    };
    let q = &p / "branch2";
    let branch2 = nn::seq_t()
        .add(nn::conv2d(
            &q / 0,
            if stride > 1 { c_in } else { branch_features },
            branch_features,
            1,
            ConvConfig {
                bias: false,
                ..Default::default()
            },
        ))
        .add(nn::batch_norm2d(
            &q / 1,
            branch_features,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(depthwise_conv(
            &q / 3,
            branch_features,
            branch_features,
            3,
            stride,
            1,
            false,
        ))
        .add(nn::batch_norm2d(
            &q / 4,
            branch_features,
            Default::default(),
        ))
        .add(nn::conv2d(
            &q / 5,
            branch_features,
            branch_features,
            1,
            ConvConfig {
                bias: false,
                ..Default::default()
            },
        ))
        .add(nn::batch_norm2d(
            &q / 6,
            branch_features,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu());

    nn::func_t(move |xs, train| {
        let ys = if stride == 1 {
            let xs = xs.chunk(2, 1);
            let (x1, x2) = (xs.get(0).unwrap(), xs.get(1).unwrap());
            Tensor::cat(&[x1, &x2.apply_t(&branch2, train)], 1)
        } else {
            Tensor::cat(
                &[xs.apply_t(&branch1, train), xs.apply_t(&branch2, train)],
                1,
            )
        };
        channel_shuffle(&ys, 2)
    })
}

fn shufflenet_v2(
    p: &nn::Path,
    stages_repeats: &[i64],
    stages_out_channels: &[i64],
    num_classes: i64,
) -> impl nn::ModuleT {
    let mut c_in = 3;
    let mut c_out = stages_out_channels[0];
    let conv1 = nn::seq_t()
        .add(nn::conv2d(
            p / "conv1" / 0,
            c_in,
            c_out,
            3,
            ConvConfig {
                stride: 2,
                padding: 1,
                bias: false,
                ..Default::default()
            },
        ))
        .add(nn::batch_norm2d(p / "conv1" / 1, c_out, Default::default()))
        .add_fn(|xs| xs.relu());

    c_in = c_out;
    let mut stages = nn::seq_t();
    for i in 2..=4 {
        let q = p / format!("stage{}", i);
        let repeats = stages_repeats[i - 2];
        let c_out = stages_out_channels[i - 1];
        let mut stage = nn::seq_t().add(inverted_residual(&q / 0, c_in, c_out, 2));

        for j in 0..repeats - 1 {
            stage = stage.add(inverted_residual(&q / (j + 1), c_out, c_out, 1));
        }
        c_in = c_out;
        stages = stages.add(stage);
    }
    c_out = *stages_out_channels.last().unwrap();
    let conv5 = nn::seq_t()
        .add(nn::conv2d(
            p / "conv5" / 0,
            c_in,
            c_out,
            1,
            ConvConfig {
                bias: false,
                ..Default::default()
            },
        ))
        .add(nn::batch_norm2d(p / "conv5" / 1, c_out, Default::default()))
        .add_fn(|xs| xs.relu());
    let fc = nn::linear(p / "fc", c_out, num_classes, Default::default());

    nn::func_t(move |xs, train| {
        xs.apply_t(&conv1, train)
            .max_pool2d(3, 2, 1, 1, false)
            .apply_t(&stages, train)
            .apply_t(&conv5, train)
            .adaptive_avg_pool2d([1, 1])
            .flat_view()
            .apply(&fc)
    })
}

pub fn shufflenet_v2_x0_5(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    shufflenet_v2(p, &[4, 8, 4], &[24, 48, 96, 192, 1024], num_classes)
}

pub fn shufflenet_v2_x1_0(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    shufflenet_v2(p, &[4, 8, 4], &[24, 116, 232, 464, 1024], num_classes)
}

pub fn shufflenet_v2_x1_5(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    shufflenet_v2(p, &[4, 8, 4], &[24, 176, 352, 704, 1024], num_classes)
}

pub fn shufflenet_v2_x2_0(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    shufflenet_v2(p, &[4, 8, 4], &[24, 244, 488, 976, 2048], num_classes)
}
