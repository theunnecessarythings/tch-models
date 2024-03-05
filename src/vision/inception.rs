use tch::{nn, IndexOp, Tensor};

fn basic_conv2d(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    ksize: i64,
    stride: i64,
    padding: i64,
) -> impl nn::ModuleT {
    nn::seq_t()
        .add(nn::conv2d(
            &p / "conv",
            c_in,
            c_out,
            ksize,
            nn::ConvConfig {
                padding,
                stride,
                bias: false,
                ..Default::default()
            },
        ))
        .add(nn::batch_norm2d(&p / "bn", c_out, Default::default()))
        .add_fn(|xs| xs.relu())
}

fn basic_conv2d2(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    ksize: [i64; 2],
    pad: [i64; 2],
) -> impl nn::ModuleT {
    let conv2d_cfg = nn::ConvConfigND::<[i64; 2]> {
        padding: pad,
        bias: false,
        ..Default::default()
    };
    let bn_cfg = nn::BatchNormConfig {
        eps: 0.001,
        ..Default::default()
    };
    nn::seq_t()
        .add(nn::conv(&p / "conv", c_in, c_out, ksize, conv2d_cfg))
        .add(nn::batch_norm2d(&p / "bn", c_out, bn_cfg))
        .add_fn(|xs| xs.relu())
}

fn max_pool2d(xs: &Tensor, ksize: i64, stride: i64) -> Tensor {
    xs.max_pool2d([ksize, ksize], [stride, stride], [0, 0], [1, 1], false)
}

fn inception_a(p: nn::Path, c_in: i64, pool_features: i64) -> impl nn::ModuleT {
    let branch1x1 = basic_conv2d(&p / "branch1x1", c_in, 64, 1, 1, 0);
    let branch5x5_1 = basic_conv2d(&p / "branch5x5_1", c_in, 48, 1, 1, 0);
    let branch5x5_2 = basic_conv2d(&p / "branch5x5_2", 48, 64, 5, 1, 2);
    let branch3x3dbl_1 = basic_conv2d(&p / "branch3x3dbl_1", c_in, 64, 1, 1, 0);
    let branch3x3dbl_2 = basic_conv2d(&p / "branch3x3dbl_2", 64, 96, 3, 1, 1);
    let branch3x3dbl_3 = basic_conv2d(&p / "branch3x3dbl_3", 96, 96, 3, 1, 1);
    let branch_pool = basic_conv2d(&p / "branch_pool", c_in, pool_features, 1, 1, 0);

    nn::func_t(move |xs, train| {
        let b1 = xs.apply_t(&branch1x1, train);
        let b2 = xs.apply_t(&branch5x5_1, train).apply_t(&branch5x5_2, train);
        let b3 = xs
            .apply_t(&branch3x3dbl_1, train)
            .apply_t(&branch3x3dbl_2, train)
            .apply_t(&branch3x3dbl_3, train);
        let b4 = xs
            .avg_pool2d(3, 1, 1, false, true, None)
            .apply_t(&branch_pool, train);
        Tensor::cat(&[b1, b2, b3, b4], 1)
    })
}

fn inception_b(p: nn::Path, c_in: i64) -> impl nn::ModuleT {
    let branch3x3 = basic_conv2d(&p / "branch3x3", c_in, 384, 3, 2, 0);
    let branch3x3dbl_1 = basic_conv2d(&p / "branch3x3dbl_1", c_in, 64, 1, 1, 0);
    let branch3x3dbl_2 = basic_conv2d(&p / "branch3x3dbl_2", 64, 96, 3, 1, 1);
    let branch3x3dbl_3 = basic_conv2d(&p / "branch3x3dbl_3", 96, 96, 3, 2, 0);

    nn::func_t(move |xs, train| {
        let b1 = xs.apply_t(&branch3x3, train);
        let b2 = xs
            .apply_t(&branch3x3dbl_1, train)
            .apply_t(&branch3x3dbl_2, train)
            .apply_t(&branch3x3dbl_3, train);
        let b3 = max_pool2d(xs, 3, 2);
        Tensor::cat(&[b1, b2, b3], 1)
    })
}

fn inception_c(p: nn::Path, c_in: i64, channels_7x7: i64) -> impl nn::ModuleT {
    let branch1x1 = basic_conv2d(&p / "branch1x1", c_in, 192, 1, 1, 0);
    let branch7x7_1 = basic_conv2d(&p / "branch7x7_1", c_in, channels_7x7, 1, 1, 0);
    let branch7x7_2 = basic_conv2d2(
        &p / "branch7x7_2",
        channels_7x7,
        channels_7x7,
        [1, 7],
        [0, 3],
    );
    let branch7x7_3 = basic_conv2d2(&p / "branch7x7_3", channels_7x7, 192, [7, 1], [3, 0]);
    let branch7x7dbl_1 = basic_conv2d(&p / "branch7x7dbl_1", c_in, channels_7x7, 1, 1, 0);
    let branch7x7dbl_2 = basic_conv2d2(
        &p / "branch7x7dbl_2",
        channels_7x7,
        channels_7x7,
        [7, 1],
        [3, 0],
    );
    let branch7x7dbl_3 = basic_conv2d2(
        &p / "branch7x7dbl_3",
        channels_7x7,
        channels_7x7,
        [1, 7],
        [0, 3],
    );
    let branch7x7dbl_4 = basic_conv2d2(
        &p / "branch7x7dbl_4",
        channels_7x7,
        channels_7x7,
        [7, 1],
        [3, 0],
    );
    let branch7x7dbl_5 = basic_conv2d2(&p / "branch7x7dbl_5", channels_7x7, 192, [1, 7], [0, 3]);
    let branch_pool = basic_conv2d(&p / "branch_pool", c_in, 192, 1, 1, 0);

    nn::func_t(move |xs, train| {
        let b1 = xs.apply_t(&branch1x1, train);
        let b2 = xs
            .apply_t(&branch7x7_1, train)
            .apply_t(&branch7x7_2, train)
            .apply_t(&branch7x7_3, train);
        let b3 = xs
            .apply_t(&branch7x7dbl_1, train)
            .apply_t(&branch7x7dbl_2, train)
            .apply_t(&branch7x7dbl_3, train)
            .apply_t(&branch7x7dbl_4, train)
            .apply_t(&branch7x7dbl_5, train);
        let b4 = xs
            .avg_pool2d([3, 3], [1, 1], [1, 1], false, true, None)
            .apply_t(&branch_pool, train);
        Tensor::cat(&[b1, b2, b3, b4], 1)
    })
}

fn inception_d(p: nn::Path, c_in: i64) -> impl nn::ModuleT {
    let branch3x3_1 = basic_conv2d(&p / "branch3x3_1", c_in, 192, 1, 1, 0);
    let branch3x3_2 = basic_conv2d(&p / "branch3x3_2", 192, 320, 3, 2, 0);
    let branch7x7x3_1 = basic_conv2d(&p / "branch7x7x3_1", c_in, 192, 1, 1, 0);
    let branch7x7x3_2 = basic_conv2d2(&p / "branch7x7x3_2", 192, 192, [1, 7], [0, 3]);
    let branch7x7x3_3 = basic_conv2d2(&p / "branch7x7x3_3", 192, 192, [7, 1], [3, 0]);
    let branch7x7x3_4 = basic_conv2d(&p / "branch7x7x3_4", 192, 192, 3, 2, 0);

    nn::func_t(move |xs, train| {
        let b1 = xs.apply_t(&branch3x3_1, train).apply_t(&branch3x3_2, train);
        let b2 = xs
            .apply_t(&branch7x7x3_1, train)
            .apply_t(&branch7x7x3_2, train)
            .apply_t(&branch7x7x3_3, train)
            .apply_t(&branch7x7x3_4, train);
        let b3 = max_pool2d(xs, 3, 2);
        Tensor::cat(&[b1, b2, b3], 1)
    })
}

fn inception_e(p: nn::Path, c_in: i64) -> impl nn::ModuleT {
    let branch1x1 = basic_conv2d(&p / "branch1x1", c_in, 320, 1, 1, 0);
    let branch3x3_1 = basic_conv2d(&p / "branch3x3_1", c_in, 384, 1, 1, 0);
    let branch3x3_2a = basic_conv2d2(&p / "branch3x3_2a", 384, 384, [1, 3], [0, 1]);
    let branch3x3_2b = basic_conv2d2(&p / "branch3x3_2b", 384, 384, [3, 1], [1, 0]);
    let branch3x3dbl_1 = basic_conv2d(&p / "branch3x3dbl_1", c_in, 448, 1, 1, 0);
    let branch3x3dbl_2 = basic_conv2d(&p / "branch3x3dbl_2", 448, 384, 3, 1, 1);
    let branch3x3dbl_3a = basic_conv2d2(&p / "branch3x3dbl_3a", 384, 384, [1, 3], [0, 1]);
    let branch3x3dbl_3b = basic_conv2d2(&p / "branch3x3dbl_3b", 384, 384, [3, 1], [1, 0]);
    let branch_pool = basic_conv2d(&p / "branch_pool", c_in, 192, 1, 1, 0);

    nn::func_t(move |xs, train| {
        let b1 = xs.apply_t(&branch1x1, train);
        let b2 = xs.apply_t(&branch3x3_1, train);
        let b2 = Tensor::cat(
            &[
                b2.apply_t(&branch3x3_2a, train),
                b2.apply_t(&branch3x3_2b, train),
            ],
            1,
        );
        let b3 = xs
            .apply_t(&branch3x3dbl_1, train)
            .apply_t(&branch3x3dbl_2, train);
        let b3 = Tensor::cat(
            &[
                b3.apply_t(&branch3x3dbl_3a, train),
                b3.apply_t(&branch3x3dbl_3b, train),
            ],
            1,
        );
        let b4 = xs
            .avg_pool2d([3, 3], [1, 1], [1, 1], false, true, None)
            .apply_t(&branch_pool, train);
        Tensor::cat(&[b1, b2, b3, b4], 1)
    })
}

fn inception_aux(p: nn::Path, c_in: i64, num_classes: i64) -> impl nn::ModuleT {
    let conv0 = basic_conv2d(&p / "conv0", c_in, 128, 1, 1, 0);
    let conv1 = basic_conv2d(&p / "conv1", 128, 768, 5, 1, 0);
    let fc = nn::linear(&p / "fc", 768, num_classes, Default::default());

    nn::func_t(move |xs, train| {
        let xs = xs.avg_pool2d([5, 5], [3, 3], [0, 0], false, true, None);
        let xs = xs.apply_t(&conv0, train);
        let xs = xs.apply_t(&conv1, train);
        let xs = xs.view([-1, 768]);
        xs.apply(&fc)
    })
}

fn inception3(
    p: &nn::Path,
    num_classes: i64,
    transform_input: bool,
    dropout: f64,
) -> impl nn::ModuleT {
    nn::seq_t()
        .add_fn(move |xs| {
            if transform_input {
                let x_ch0 = xs.i((.., 0)).unsqueeze(1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5;
                let x_ch1 = xs.i((.., 1)).unsqueeze(1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5;
                let x_ch2 = xs.i((.., 2)).unsqueeze(1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5;
                Tensor::cat(&[x_ch0, x_ch1, x_ch2], 1)
            } else {
                xs.shallow_clone()
            }
        })
        .add(basic_conv2d(p / "Conv2d_1a_3x3", 3, 32, 3, 2, 0))
        .add(basic_conv2d(p / "Conv2d_2a_3x3", 32, 32, 3, 1, 0))
        .add(basic_conv2d(p / "Conv2d_2b_3x3", 32, 64, 3, 1, 1))
        .add_fn(|xs| max_pool2d(&xs.relu(), 3, 2))
        .add(basic_conv2d(p / "Conv2d_3b_1x1", 64, 80, 1, 1, 0))
        .add(basic_conv2d(p / "Conv2d_4a_3x3", 80, 192, 3, 1, 0))
        .add_fn(|xs| max_pool2d(&xs.relu(), 3, 2))
        .add(inception_a(p / "Mixed_5b", 192, 32))
        .add(inception_a(p / "Mixed_5c", 256, 64))
        .add(inception_a(p / "Mixed_5d", 288, 64))
        .add(inception_b(p / "Mixed_6a", 288))
        .add(inception_c(p / "Mixed_6b", 768, 128))
        .add(inception_c(p / "Mixed_6c", 768, 160))
        .add(inception_c(p / "Mixed_6d", 768, 160))
        .add(inception_c(p / "Mixed_6e", 768, 192))
        .add(inception_d(p / "Mixed_7a", 768))
        .add(inception_e(p / "Mixed_7b", 1280))
        .add(inception_e(p / "Mixed_7c", 2048))
        .add_fn_t(move |xs, train| {
            xs.adaptive_avg_pool2d([1, 1])
                .dropout(dropout, train)
                .flat_view()
        })
        .add(nn::linear(p / "fc", 2048, num_classes, Default::default()))
}

pub fn inception_v3(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    inception3(p, num_classes, true, 0.5)
}
