use anyhow::Result;
use tch::{
    nn::{self, VarStore},
    vision::imagenet,
    Device, Kind, Tensor,
};
mod alexnet;
mod convnext;
mod densenet;
mod efficientnet;
mod googlenet;
mod inception;
mod maxvit;
mod mnasnet;
mod mobilenet_v2;
mod mobilenet_v3;
mod regnet;
mod resnet;
mod shufflenet_v2;
mod squeezenet;
mod swin_transformer;
mod util;
mod vgg;
mod vit;

#[derive(Debug)]
enum Model {
    AlexNet,
    ConvNextBase,
    ConvNextSmall,
    ConvNextTiny,
    ConvNextLarge,
    DenseNet121,
    DenseNet161,
    DenseNet169,
    DenseNet201,
    GoogleNet,
    InceptionV3,
    MnasNet0_5,
    MnasNet0_75,
    MnasNet1_0,
    MnasNet1_3,
    MobileNetV2,
    MobileNetV3Small,
    MobileNetV3Large,
    RegNetY400MF,
    RegNetY800MF,
    RegNetY1_6GF,
    RegNetY3_2GF,
    RegNetY8GF,
    RegNetY16GF,
    RegNetY32GF,
    RegNetY128GF,
    RegNetX400MF,
    RegNetX800MF,
    RegNetX1_6GF,
    RegNetX3_2GF,
    RegNetX8GF,
    RegNetX16GF,
    RegNetX32GF,
    ShuffleNetV2X0_5,
    ShuffleNetV2X1_0,
    ShuffleNetV2X1_5,
    ShuffleNetV2X2_0,
    SqueezeNet1_0,
    SqueezeNet1_1,
    VGG11,
    VGG11BN,
    VGG13,
    VGG13BN,
    VGG16,
    VGG16BN,
    VGG19,
    VGG19BN,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNext50_32x4d,
    ResNext101_32x8d,
    ResNext101_64x4d,
    WideResNet50_2,
    WideResNet101_2,
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
    EfficientNetV2S,
    EfficientNetV2M,
    EfficientNetV2L,
    ViTB16,
    ViTB32,
    ViTBH14,
    ViTL16,
    ViTL32,
    MaxViTT,
    SwinT,
    SwinS,
    SwinB,
    SwinV2T,
    SwinV2S,
    SwinV2B,
}

fn load_model(model: Model) -> Result<(Box<dyn nn::ModuleT>, VarStore)> {
    let mut vs = VarStore::new(Device::cuda_if_available());
    let m: Box<dyn nn::ModuleT> = match model {
        Model::AlexNet => Box::new(alexnet::alexnet(&vs.root(), 1000)),
        Model::ConvNextBase => Box::new(convnext::convnext_base(&vs.root(), 1000)),
        Model::ConvNextSmall => Box::new(convnext::convnext_small(&vs.root(), 1000)),
        Model::ConvNextTiny => Box::new(convnext::convnext_tiny(&vs.root(), 1000)),
        Model::ConvNextLarge => Box::new(convnext::convnext_large(&vs.root(), 1000)),
        Model::DenseNet121 => Box::new(densenet::densenet121(&vs.root(), 1000)),
        Model::DenseNet161 => Box::new(densenet::densenet161(&vs.root(), 1000)),
        Model::DenseNet169 => Box::new(densenet::densenet169(&vs.root(), 1000)),
        Model::DenseNet201 => Box::new(densenet::densenet201(&vs.root(), 1000)),
        Model::GoogleNet => Box::new(googlenet::googlenet(&vs.root(), 1000)),
        Model::InceptionV3 => Box::new(inception::inception_v3(&vs.root(), 1000)),
        Model::MnasNet0_5 => Box::new(mnasnet::mnasnet0_5(&vs.root(), 1000)),
        Model::MnasNet0_75 => Box::new(mnasnet::mnasnet0_75(&vs.root(), 1000)),
        Model::MnasNet1_0 => Box::new(mnasnet::mnasnet1_0(&vs.root(), 1000)),
        Model::MnasNet1_3 => Box::new(mnasnet::mnasnet1_3(&vs.root(), 1000)),
        Model::MobileNetV2 => Box::new(mobilenet_v2::mobilenet_v2(&vs.root(), 1000)),
        Model::MobileNetV3Small => Box::new(mobilenet_v3::mobilenet_v3_small(&vs.root(), 1000)),
        Model::MobileNetV3Large => Box::new(mobilenet_v3::mobilenet_v3_large(&vs.root(), 1000)),
        Model::RegNetY400MF => Box::new(regnet::regnet_y_400mf(&vs.root(), 1000)),
        Model::RegNetY800MF => Box::new(regnet::regnet_y_800mf(&vs.root(), 1000)),
        Model::RegNetY1_6GF => Box::new(regnet::regnet_y_1_6gf(&vs.root(), 1000)),
        Model::RegNetY3_2GF => Box::new(regnet::regnet_y_3_2gf(&vs.root(), 1000)),
        Model::RegNetY8GF => Box::new(regnet::regnet_y_8gf(&vs.root(), 1000)),
        Model::RegNetY16GF => Box::new(regnet::regnet_y_16gf(&vs.root(), 1000)),
        Model::RegNetY32GF => Box::new(regnet::regnet_y_32gf(&vs.root(), 1000)),
        Model::RegNetY128GF => Box::new(regnet::regnet_y_128gf(&vs.root(), 1000)),
        Model::RegNetX400MF => Box::new(regnet::regnet_x_400mf(&vs.root(), 1000)),
        Model::RegNetX800MF => Box::new(regnet::regnet_x_800mf(&vs.root(), 1000)),
        Model::RegNetX1_6GF => Box::new(regnet::regnet_x_1_6gf(&vs.root(), 1000)),
        Model::RegNetX3_2GF => Box::new(regnet::regnet_x_3_2gf(&vs.root(), 1000)),
        Model::RegNetX8GF => Box::new(regnet::regnet_x_8gf(&vs.root(), 1000)),
        Model::RegNetX16GF => Box::new(regnet::regnet_x_16gf(&vs.root(), 1000)),
        Model::RegNetX32GF => Box::new(regnet::regnet_x_32gf(&vs.root(), 1000)),
        Model::ShuffleNetV2X0_5 => Box::new(shufflenet_v2::shufflenet_v2_x0_5(&vs.root(), 1000)),
        Model::ShuffleNetV2X1_0 => Box::new(shufflenet_v2::shufflenet_v2_x1_0(&vs.root(), 1000)),
        Model::ShuffleNetV2X1_5 => Box::new(shufflenet_v2::shufflenet_v2_x1_5(&vs.root(), 1000)),
        Model::ShuffleNetV2X2_0 => Box::new(shufflenet_v2::shufflenet_v2_x2_0(&vs.root(), 1000)),
        Model::SqueezeNet1_0 => Box::new(squeezenet::squeezenet1_0(&vs.root(), 1000)),
        Model::SqueezeNet1_1 => Box::new(squeezenet::squeezenet1_1(&vs.root(), 1000)),
        Model::VGG11 => Box::new(vgg::vgg11(&vs.root(), 1000)),
        Model::VGG11BN => Box::new(vgg::vgg11_bn(&vs.root(), 1000)),
        Model::VGG13 => Box::new(vgg::vgg13(&vs.root(), 1000)),
        Model::VGG13BN => Box::new(vgg::vgg13_bn(&vs.root(), 1000)),
        Model::VGG16 => Box::new(vgg::vgg16(&vs.root(), 1000)),
        Model::VGG16BN => Box::new(vgg::vgg16_bn(&vs.root(), 1000)),
        Model::VGG19 => Box::new(vgg::vgg19(&vs.root(), 1000)),
        Model::VGG19BN => Box::new(vgg::vgg19_bn(&vs.root(), 1000)),
        Model::ResNet18 => Box::new(resnet::resnet18(&vs.root(), 1000)),
        Model::ResNet34 => Box::new(resnet::resnet34(&vs.root(), 1000)),
        Model::ResNet50 => Box::new(resnet::resnet50(&vs.root(), 1000)),
        Model::ResNet101 => Box::new(resnet::resnet101(&vs.root(), 1000)),
        Model::ResNet152 => Box::new(resnet::resnet152(&vs.root(), 1000)),
        Model::ResNext50_32x4d => Box::new(resnet::resnext50_32x4d(&vs.root(), 1000)),
        Model::ResNext101_32x8d => Box::new(resnet::resnext101_32x8d(&vs.root(), 1000)),
        Model::ResNext101_64x4d => Box::new(resnet::resnext101_64x4d(&vs.root(), 1000)),
        Model::WideResNet50_2 => Box::new(resnet::wide_resnet50_2(&vs.root(), 1000)),
        Model::WideResNet101_2 => Box::new(resnet::wide_resnet101_2(&vs.root(), 1000)),
        Model::EfficientNetB0 => Box::new(efficientnet::efficientnet_b0(&vs.root(), 1000)),
        Model::EfficientNetB1 => Box::new(efficientnet::efficientnet_b1(&vs.root(), 1000)),
        Model::EfficientNetB2 => Box::new(efficientnet::efficientnet_b2(&vs.root(), 1000)),
        Model::EfficientNetB3 => Box::new(efficientnet::efficientnet_b3(&vs.root(), 1000)),
        Model::EfficientNetB4 => Box::new(efficientnet::efficientnet_b4(&vs.root(), 1000)),
        Model::EfficientNetB5 => Box::new(efficientnet::efficientnet_b5(&vs.root(), 1000)),
        Model::EfficientNetB6 => Box::new(efficientnet::efficientnet_b6(&vs.root(), 1000)),
        Model::EfficientNetB7 => Box::new(efficientnet::efficientnet_b7(&vs.root(), 1000)),
        Model::EfficientNetV2S => Box::new(efficientnet::efficientnet_v2s(&vs.root(), 1000)),
        Model::EfficientNetV2M => Box::new(efficientnet::efficientnet_v2m(&vs.root(), 1000)),
        Model::EfficientNetV2L => Box::new(efficientnet::efficientnet_v2l(&vs.root(), 1000)), // TODO:
        // Known Issue : Incorrect output
        Model::ViTB16 => Box::new(vit::vit_b_16(&vs.root(), 1000)),
        Model::ViTB32 => Box::new(vit::vit_b_32(&vs.root(), 1000)),
        Model::ViTBH14 => Box::new(vit::vit_h_14(&vs.root(), 1000)),
        Model::ViTL16 => Box::new(vit::vit_l_16(&vs.root(), 1000)),
        Model::ViTL32 => Box::new(vit::vit_l_32(&vs.root(), 1000)),
        Model::MaxViTT => Box::new(maxvit::maxvit_t(&vs.root(), 1000)),
        Model::SwinT => Box::new(swin_transformer::swin_t(&vs.root(), 1000)),
        Model::SwinS => Box::new(swin_transformer::swin_s(&vs.root(), 1000)),
        Model::SwinB => Box::new(swin_transformer::swin_b(&vs.root(), 1000)),
        Model::SwinV2T => Box::new(swin_transformer::swin_v2_t(&vs.root(), 1000)),
        Model::SwinV2S => Box::new(swin_transformer::swin_v2_s(&vs.root(), 1000)),
        Model::SwinV2B => Box::new(swin_transformer::swin_v2_b(&vs.root(), 1000)),
    };

    match model {
        Model::AlexNet => vs.load("alexnet.safetensors")?,
        Model::ConvNextBase => vs.load("convnext_base.safetensors")?,
        Model::ConvNextSmall => vs.load("convnext_small.safetensors")?,
        Model::ConvNextTiny => vs.load("convnext_tiny.safetensors")?,
        Model::ConvNextLarge => vs.load("convnext_large.safetensors")?,
        Model::DenseNet121 => vs.load("densenet121.safetensors")?,
        Model::DenseNet161 => vs.load("densenet161.safetensors")?,
        Model::DenseNet169 => vs.load("densenet169.safetensors")?,
        Model::DenseNet201 => vs.load("densenet201.safetensors")?,
        Model::GoogleNet => vs.load("googlenet.safetensors")?,
        Model::InceptionV3 => vs.load("inception_v3.safetensors")?,
        Model::MnasNet0_5 => vs.load("mnasnet0_5.safetensors")?,
        Model::MnasNet0_75 => vs.load("mnasnet0_75.safetensors")?,
        Model::MnasNet1_0 => vs.load("mnasnet1_0.safetensors")?,
        Model::MnasNet1_3 => vs.load("mnasnet1_3.safetensors")?,
        Model::MobileNetV2 => vs.load("mobilenet_v2.safetensors")?,
        Model::MobileNetV3Small => vs.load("mobilenet_v3_small.safetensors")?,
        Model::MobileNetV3Large => vs.load("mobilenet_v3_large.safetensors")?,
        Model::RegNetY400MF => vs.load("regnet_y_400mf.safetensors")?,
        Model::RegNetY800MF => vs.load("regnet_y_800mf.safetensors")?,
        Model::RegNetY1_6GF => vs.load("regnet_y_1_6gf.safetensors")?,
        Model::RegNetY3_2GF => vs.load("regnet_y_3_2gf.safetensors")?,
        Model::RegNetY8GF => vs.load("regnet_y_8gf.safetensors")?,
        Model::RegNetY16GF => vs.load("regnet_y_16gf.safetensors")?,
        Model::RegNetY32GF => vs.load("regnet_y_32gf.safetensors")?,
        Model::RegNetY128GF => vs.load("regnet_y_128gf.safetensors")?,
        Model::RegNetX400MF => vs.load("regnet_x_400mf.safetensors")?,
        Model::RegNetX800MF => vs.load("regnet_x_800mf.safetensors")?,
        Model::RegNetX1_6GF => vs.load("regnet_x_1_6gf.safetensors")?,
        Model::RegNetX3_2GF => vs.load("regnet_x_3_2gf.safetensors")?,
        Model::RegNetX8GF => vs.load("regnet_x_8gf.safetensors")?,
        Model::RegNetX16GF => vs.load("regnet_x_16gf.safetensors")?,
        Model::RegNetX32GF => vs.load("regnet_x_32gf.safetensors")?,
        Model::ShuffleNetV2X0_5 => vs.load("shufflenet_v2_x0_5.safetensors")?,
        Model::ShuffleNetV2X1_0 => vs.load("shufflenet_v2_x1_0.safetensors")?,
        Model::ShuffleNetV2X1_5 => vs.load("shufflenet_v2_x1_5.safetensors")?,
        Model::ShuffleNetV2X2_0 => vs.load("shufflenet_v2_x2_0.safetensors")?,
        Model::SqueezeNet1_0 => vs.load("squeezenet1_0.safetensors")?,
        Model::SqueezeNet1_1 => vs.load("squeezenet1_1.safetensors")?,
        Model::VGG11 => vs.load("vgg11.safetensors")?,
        Model::VGG11BN => vs.load("vgg11_bn.safetensors")?,
        Model::VGG13 => vs.load("vgg13.safetensors")?,
        Model::VGG13BN => vs.load("vgg13_bn.safetensors")?,
        Model::VGG16 => vs.load("vgg16.safetensors")?,
        Model::VGG16BN => vs.load("vgg16_bn.safetensors")?,
        Model::VGG19 => vs.load("vgg19.safetensors")?,
        Model::VGG19BN => vs.load("vgg19_bn.safetensors")?,
        Model::ResNet18 => vs.load("resnet18.safetensors")?,
        Model::ResNet34 => vs.load("resnet34.safetensors")?,
        Model::ResNet50 => vs.load("resnet50.safetensors")?,
        Model::ResNet101 => vs.load("resnet101.safetensors")?,
        Model::ResNet152 => vs.load("resnet152.safetensors")?,
        Model::ResNext50_32x4d => vs.load("resnext50_32x4d.safetensors")?,
        Model::ResNext101_32x8d => vs.load("resnext101_32x8d.safetensors")?,
        Model::ResNext101_64x4d => vs.load("resnext101_64x4d.safetensors")?,
        Model::WideResNet50_2 => vs.load("wide_resnet50_2.safetensors")?,
        Model::WideResNet101_2 => vs.load("wide_resnet101_2.safetensors")?,
        Model::EfficientNetB0 => vs.load("efficientnet_b0.safetensors")?,
        Model::EfficientNetB1 => vs.load("efficientnet_b1.safetensors")?,
        Model::EfficientNetB2 => vs.load("efficientnet_b2.safetensors")?,
        Model::EfficientNetB3 => vs.load("efficientnet_b3.safetensors")?,
        Model::EfficientNetB4 => vs.load("efficientnet_b4.safetensors")?,
        Model::EfficientNetB5 => vs.load("efficientnet_b5.safetensors")?,
        Model::EfficientNetB6 => vs.load("efficientnet_b6.safetensors")?,
        Model::EfficientNetB7 => vs.load("efficientnet_b7.safetensors")?,
        Model::EfficientNetV2S => vs.load("efficientnet_v2_s.safetensors")?,
        Model::EfficientNetV2M => vs.load("efficientnet_v2_m.safetensors")?,
        Model::EfficientNetV2L => vs.load("efficientnet_v2_l.safetensors")?,
        Model::ViTB16 => vs.load("vit_b_16.safetensors")?,
        Model::ViTB32 => vs.load("vit_b_32.safetensors")?,
        Model::ViTBH14 => vs.load("vit_h_14.safetensors")?,
        Model::ViTL16 => vs.load("vit_l_16.safetensors")?,
        Model::ViTL32 => vs.load("vit_l_32.safetensors")?,
        Model::MaxViTT => vs.load("maxvit_t.safetensors")?,
        Model::SwinT => vs.load("swin_t.safetensors")?,
        Model::SwinS => vs.load("swin_s.safetensors")?,
        Model::SwinB => vs.load("swin_b.safetensors")?,
        Model::SwinV2T => vs.load("swin_v2_t.safetensors")?,
        Model::SwinV2S => vs.load("swin_v2_s.safetensors")?,
        Model::SwinV2B => vs.load("swin_v2_b.safetensors")?,
    }

    Ok((m, vs))
}

fn print_varstore(vs: VarStore) {
    let named_variables: Vec<(String, Vec<i64>)> = vs
        .variables_
        .lock()
        .unwrap()
        .named_variables
        .iter()
        .map(|(key, val)| (key.clone(), val.size()))
        .collect();

    // println!("{:#?}", named_variables);
    for (key, shape) in named_variables {
        println!("{:?} {:?}", key, shape);
    }
}

fn main() -> Result<()> {
    for model in [
        // Model::AlexNet,
        // Model::ConvNextBase,
        // Model::ConvNextSmall,
        // Model::ConvNextTiny,
        // Model::ConvNextLarge,
        // Model::DenseNet121,
        // Model::DenseNet161,
        // Model::DenseNet169,
        // Model::DenseNet201,
        // Model::GoogleNet,
        // Model::InceptionV3,
        // Model::MnasNet0_5,
        // Model::MnasNet0_75,
        // Model::MnasNet1_0,
        // Model::MnasNet1_3,
        // Model::MobileNetV2,
        // Model::MobileNetV3Small,
        // Model::MobileNetV3Large,
        // Model::RegNetY400MF,
        // Model::RegNetY800MF,
        // Model::RegNetY1_6GF,
        // Model::RegNetY3_2GF,
        // Model::RegNetY8GF,
        // Model::RegNetY16GF,
        // Model::RegNetY32GF,
        // Model::RegNetY128GF,
        // Model::RegNetX400MF,
        // Model::RegNetX800MF,
        // Model::RegNetX1_6GF,
        // Model::RegNetX3_2GF,
        // Model::RegNetX8GF,
        // Model::RegNetX16GF,
        // Model::RegNetX32GF,
        // Model::ShuffleNetV2X0_5,
        // Model::ShuffleNetV2X1_0,
        // Model::ShuffleNetV2X1_5,
        // Model::ShuffleNetV2X2_0,
        // Model::SqueezeNet1_0,
        // Model::SqueezeNet1_1,
        // Model::VGG11,
        // Model::VGG11BN,
        // Model::VGG13,
        // Model::VGG13BN,
        // Model::VGG16,
        // Model::VGG16BN,
        // Model::VGG19,
        // Model::VGG19BN,
        // Model::ResNet18,
        // Model::ResNet34,
        // Model::ResNet50,
        // Model::ResNet101,
        // Model::ResNet152,
        // Model::ResNext50_32x4d,
        // Model::ResNext101_32x8d,
        // Model::ResNext101_64x4d,
        // Model::WideResNet50_2,
        // Model::WideResNet101_2,
        // Model::EfficientNetB0,
        // Model::EfficientNetB1,
        // Model::EfficientNetB2,
        // Model::EfficientNetB3,
        // Model::EfficientNetB4,
        // Model::EfficientNetB5,
        // Model::EfficientNetB6,
        // Model::EfficientNetB7,
        // Model::EfficientNetV2S,
        // Model::EfficientNetV2M,
        // Model::EfficientNetV2L,
        // Model::ViTB16,
        // Model::ViTB32,
        // Model::ViTBH14,
        // Model::ViTL16,
        // Model::ViTL32,
        // Model::MaxViTT,
        Model::SwinT,
        // Model::SwinS,
        // Model::SwinB,
        Model::SwinV2T,
        // Model::SwinV2S,
        // Model::SwinV2B,
    ] {
        println!("Loading model: {:?}", model);

        let (model, vs) = load_model(model)?;
        let image = imagenet::load_image_and_resize224("../candle-test/examples/tiger.jpg")?
            .to_device(vs.device())
            .unsqueeze(0);
        // let image = Tensor::ones([1, 3, 224, 224], (Kind::Float, vs.device()));

        // let w = &vs.variables_.lock().unwrap().named_variables;
        // let w = w.get("features.6.1.block.3.1.running_mean").unwrap();
        // println!("{w}");

        let output = model.forward_t(&image, false).softmax(-1, Kind::Float);

        for (probability, class) in imagenet::top(&output, 5).iter() {
            println!("{:50} {:5.2}%", class, 100.0 * probability)
        }
        println!();

        // print_varstore(vs);
    }

    Ok(())
}
