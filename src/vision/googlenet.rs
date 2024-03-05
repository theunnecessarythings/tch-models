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

fn inception(
    p: nn::Path,
    c_in: i64,
    ch1x1: i64,
    ch3x3red: i64,
    ch3x3: i64,
    ch5x5red: i64,
    ch5x5: i64,
    pool_proj: i64,
) -> impl nn::ModuleT {
    let branch1 = basic_conv2d(&p / "branch1", c_in, ch1x1, 1, 1, 0);
    let branch2 = nn::seq_t()
        .add(basic_conv2d(&p / "branch2" / 0, c_in, ch3x3red, 1, 1, 0))
        .add(basic_conv2d(&p / "branch2" / 1, ch3x3red, ch3x3, 3, 1, 1));
    let branch3 = nn::seq_t()
        .add(basic_conv2d(&p / "branch3" / 0, c_in, ch5x5red, 1, 1, 0))
        .add(basic_conv2d(&p / "branch3" / 1, ch5x5red, ch5x5, 3, 1, 1));
    let branch4 = nn::seq_t()
        .add_fn(|xs| xs.max_pool2d(3, 1, 1, 1, true))
        .add(basic_conv2d(&p / "branch4" / 1, c_in, pool_proj, 1, 1, 0));
    nn::func_t(move |xs, train| {
        let b1 = xs.apply_t(&branch1, train);
        let b2 = xs.apply_t(&branch2, train);
        let b3 = xs.apply_t(&branch3, train);
        let b4 = xs.apply_t(&branch4, train);
        Tensor::cat(&[b1, b2, b3, b4], 1)
    })
}

fn inception_aux(p: nn::Path, c_in: i64, num_classes: i64, dropout: f64) -> impl nn::ModuleT {
    nn::seq_t()
        .add_fn(|xs| xs.adaptive_avg_pool1d([4, 4]))
        .add(basic_conv2d(&p / "conv", c_in, 128, 1, 1, 0))
        .add(nn::linear(&p / "fc1", 2048, 1024, Default::default()))
        .add_fn_t(move |xs, train| xs.flatten(1, -1).relu().dropout(dropout, train))
        .add(nn::linear(
            &p / "fc2",
            1024,
            num_classes,
            Default::default(),
        ))
}
fn _googlenet(
    p: &nn::Path,
    num_classes: i64,
    aux_logits: bool,
    transform_input: bool,
    dropout: f64,
    dropout_aux: f64,
) -> impl nn::ModuleT {
    let conv1 = basic_conv2d(p / "conv1", 3, 64, 7, 2, 3);
    let conv2 = basic_conv2d(p / "conv2", 64, 64, 1, 1, 0);
    let conv3 = basic_conv2d(p / "conv3", 64, 192, 3, 1, 1);
    let inception3a = inception(p / "inception3a", 192, 64, 96, 128, 16, 32, 32);
    let inception3b = inception(p / "inception3b", 256, 128, 128, 192, 32, 96, 64);
    let inception4a = inception(p / "inception4a", 480, 192, 96, 208, 16, 48, 64);
    let inception4b = inception(p / "inception4b", 512, 160, 112, 224, 24, 64, 64);
    let inception4c = inception(p / "inception4c", 512, 128, 128, 256, 24, 64, 64);
    let inception4d = inception(p / "inception4d", 512, 112, 144, 288, 32, 64, 64);
    let inception4e = inception(p / "inception4e", 528, 256, 160, 320, 32, 128, 128);
    let inception5a = inception(p / "inception5a", 832, 256, 160, 320, 32, 128, 128);
    let inception5b = inception(p / "inception5b", 832, 384, 192, 384, 48, 128, 128);

    let aux1 = if aux_logits {
        Some(inception_aux(p / "aux1", 512, num_classes, dropout_aux))
    } else {
        None
    };
    let aux2 = if aux_logits {
        Some(inception_aux(p / "aux2", 528, num_classes, dropout_aux))
    } else {
        None
    };

    let fc = nn::linear(p / "fc", 1024, num_classes, Default::default());

    nn::func_t(move |xs, train| {
        let mut xs = xs.shallow_clone();
        if transform_input {
            let x_ch0 = xs.i((.., 0)).unsqueeze(1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5;
            let x_ch1 = xs.i((.., 1)).unsqueeze(1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5;
            let x_ch2 = xs.i((.., 2)).unsqueeze(1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5;
            xs = Tensor::cat(&[x_ch0, x_ch1, x_ch2], 1);
        }

        xs = xs
            .apply_t(&conv1, train)
            .max_pool2d(3, 2, 0, 1, true)
            .apply_t(&conv2, train)
            .apply_t(&conv3, train)
            .max_pool2d(3, 2, 0, 1, true)
            .apply_t(&inception3a, train)
            .apply_t(&inception3b, train)
            .max_pool2d(3, 2, 0, 1, true)
            .apply_t(&inception4a, train);
        let mut aux1_out = None;
        if train && aux1.is_some() {
            aux1_out = Some(xs.apply_t(aux1.as_ref().unwrap(), train));
        }
        xs = xs
            .apply_t(&inception4b, train)
            .apply_t(&inception4c, train)
            .apply_t(&inception4d, train);
        let mut aux2_out = None;
        if train && aux2.is_some() {
            aux2_out = Some(xs.apply_t(aux2.as_ref().unwrap(), train));
        }
        xs = xs
            .apply_t(&inception4e, train)
            .max_pool2d(2, 2, 0, 1, true)
            .apply_t(&inception5a, train)
            .apply_t(&inception5b, train)
            .adaptive_avg_pool2d([1, 1])
            .flatten(1, -1)
            .dropout(dropout, train)
            .apply(&fc);

        // (xs, aux2_out, aux1_out)
        xs
    })
}

pub fn googlenet(p: &nn::Path, num_classes: i64) -> impl nn::ModuleT {
    _googlenet(p, num_classes, false, true, 0.2, 0.7)
}
