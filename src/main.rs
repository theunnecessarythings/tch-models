use std::{
    io,
    path::{Path, PathBuf},
};

use ai_dataloader::{indexable::DataLoader, Len};
use anyhow::Result;
use image::imageops::FilterType;
use imagenet::Transforms;
use tch::{
    kind,
    nn::{self, VarStore},
    vision::{
        dataset::Dataset,
        imagenet::{load_image, load_image_and_resize224},
    },
    Device, Kind, TchError, Tensor,
};
use indicatif::{ProgressIterator, ProgressStyle};
mod alexnet;
mod convnext;
mod densenet;
mod efficientnet;
mod googlenet;
mod imagenet;
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

use ai_dataloader::collate::TorchCollate;



#[derive(Debug, Clone)]
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

fn load_model(model: Model) -> Result<(Box<dyn nn::ModuleT>, VarStore, Transforms)> {
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
        Model::AlexNet => vs.load("safetensors/alexnet.safetensors")?,
        Model::ConvNextBase => vs.load("safetensors/convnext_base.safetensors")?,
        Model::ConvNextSmall => vs.load("safetensors/convnext_small.safetensors")?,
        Model::ConvNextTiny => vs.load("safetensors/convnext_tiny.safetensors")?,
        Model::ConvNextLarge => vs.load("safetensors/convnext_large.safetensors")?,
        Model::DenseNet121 => vs.load("safetensors/densenet121.safetensors")?,
        Model::DenseNet161 => vs.load("safetensors/densenet161.safetensors")?,
        Model::DenseNet169 => vs.load("safetensors/densenet169.safetensors")?,
        Model::DenseNet201 => vs.load("safetensors/densenet201.safetensors")?,
        Model::GoogleNet => vs.load("safetensors/googlenet.safetensors")?,
        Model::InceptionV3 => vs.load("safetensors/inception_v3.safetensors")?,
        Model::MnasNet0_5 => vs.load("safetensors/mnasnet0_5.safetensors")?,
        Model::MnasNet0_75 => vs.load("safetensors/mnasnet0_75.safetensors")?,
        Model::MnasNet1_0 => vs.load("safetensors/mnasnet1_0.safetensors")?,
        Model::MnasNet1_3 => vs.load("safetensors/mnasnet1_3.safetensors")?,
        Model::MobileNetV2 => vs.load("safetensors/mobilenet_v2.safetensors")?,
        Model::MobileNetV3Small => vs.load("safetensors/mobilenet_v3_small.safetensors")?,
        Model::MobileNetV3Large => vs.load("safetensors/mobilenet_v3_large.safetensors")?,
        Model::RegNetY400MF => vs.load("safetensors/regnet_y_400mf.safetensors")?,
        Model::RegNetY800MF => vs.load("safetensors/regnet_y_800mf.safetensors")?,
        Model::RegNetY1_6GF => vs.load("safetensors/regnet_y_1_6gf.safetensors")?,
        Model::RegNetY3_2GF => vs.load("safetensors/regnet_y_3_2gf.safetensors")?,
        Model::RegNetY8GF => vs.load("safetensors/regnet_y_8gf.safetensors")?,
        Model::RegNetY16GF => vs.load("safetensors/regnet_y_16gf.safetensors")?,
        Model::RegNetY32GF => vs.load("safetensors/regnet_y_32gf.safetensors")?,
        Model::RegNetY128GF => vs.load("safetensors/regnet_y_128gf.safetensors")?,
        Model::RegNetX400MF => vs.load("safetensors/regnet_x_400mf.safetensors")?,
        Model::RegNetX800MF => vs.load("safetensors/regnet_x_800mf.safetensors")?,
        Model::RegNetX1_6GF => vs.load("safetensors/regnet_x_1_6gf.safetensors")?,
        Model::RegNetX3_2GF => vs.load("safetensors/regnet_x_3_2gf.safetensors")?,
        Model::RegNetX8GF => vs.load("safetensors/regnet_x_8gf.safetensors")?,
        Model::RegNetX16GF => vs.load("safetensors/regnet_x_16gf.safetensors")?,
        Model::RegNetX32GF => vs.load("safetensors/regnet_x_32gf.safetensors")?,
        Model::ShuffleNetV2X0_5 => vs.load("safetensors/shufflenet_v2_x0_5.safetensors")?,
        Model::ShuffleNetV2X1_0 => vs.load("safetensors/shufflenet_v2_x1_0.safetensors")?,
        Model::ShuffleNetV2X1_5 => vs.load("safetensors/shufflenet_v2_x1_5.safetensors")?,
        Model::ShuffleNetV2X2_0 => vs.load("safetensors/shufflenet_v2_x2_0.safetensors")?,
        Model::SqueezeNet1_0 => vs.load("safetensors/squeezenet1_0.safetensors")?,
        Model::SqueezeNet1_1 => vs.load("safetensors/squeezenet1_1.safetensors")?,
        Model::VGG11 => vs.load("safetensors/vgg11.safetensors")?,
        Model::VGG11BN => vs.load("safetensors/vgg11_bn.safetensors")?,
        Model::VGG13 => vs.load("safetensors/vgg13.safetensors")?,
        Model::VGG13BN => vs.load("safetensors/vgg13_bn.safetensors")?,
        Model::VGG16 => vs.load("safetensors/vgg16.safetensors")?,
        Model::VGG16BN => vs.load("safetensors/vgg16_bn.safetensors")?,
        Model::VGG19 => vs.load("safetensors/vgg19.safetensors")?,
        Model::VGG19BN => vs.load("safetensors/vgg19_bn.safetensors")?,
        Model::ResNet18 => vs.load("safetensors/resnet18.safetensors")?,
        Model::ResNet34 => vs.load("safetensors/resnet34.safetensors")?,
        Model::ResNet50 => vs.load("safetensors/resnet50.safetensors")?,
        Model::ResNet101 => vs.load("safetensors/resnet101.safetensors")?,
        Model::ResNet152 => vs.load("safetensors/resnet152.safetensors")?,
        Model::ResNext50_32x4d => vs.load("safetensors/resnext50_32x4d.safetensors")?,
        Model::ResNext101_32x8d => vs.load("safetensors/resnext101_32x8d.safetensors")?,
        Model::ResNext101_64x4d => vs.load("safetensors/resnext101_64x4d.safetensors")?,
        Model::WideResNet50_2 => vs.load("safetensors/wide_resnet50_2.safetensors")?,
        Model::WideResNet101_2 => vs.load("safetensors/wide_resnet101_2.safetensors")?,
        Model::EfficientNetB0 => vs.load("safetensors/efficientnet_b0.safetensors")?,
        Model::EfficientNetB1 => vs.load("safetensors/efficientnet_b1.safetensors")?,
        Model::EfficientNetB2 => vs.load("safetensors/efficientnet_b2.safetensors")?,
        Model::EfficientNetB3 => vs.load("safetensors/efficientnet_b3.safetensors")?,
        Model::EfficientNetB4 => vs.load("safetensors/efficientnet_b4.safetensors")?,
        Model::EfficientNetB5 => vs.load("safetensors/efficientnet_b5.safetensors")?,
        Model::EfficientNetB6 => vs.load("safetensors/efficientnet_b6.safetensors")?,
        Model::EfficientNetB7 => vs.load("safetensors/efficientnet_b7.safetensors")?,
        Model::EfficientNetV2S => vs.load("safetensors/efficientnet_v2_s.safetensors")?,
        Model::EfficientNetV2M => vs.load("safetensors/efficientnet_v2_m.safetensors")?,
        Model::EfficientNetV2L => vs.load("safetensors/efficientnet_v2_l.safetensors")?,
        Model::ViTB16 => vs.load("safetensors/vit_b_16.safetensors")?,
        Model::ViTB32 => vs.load("safetensors/vit_b_32.safetensors")?,
        Model::ViTBH14 => vs.load("safetensors/vit_h_14.safetensors")?,
        Model::ViTL16 => vs.load("safetensors/vit_l_16.safetensors")?,
        Model::ViTL32 => vs.load("safetensors/vit_l_32.safetensors")?,
        Model::MaxViTT => vs.load("safetensors/maxvit_t.safetensors")?,
        Model::SwinT => vs.load("safetensors/swin_t.safetensors")?,
        Model::SwinS => vs.load("safetensors/swin_s.safetensors")?,
        Model::SwinB => vs.load("safetensors/swin_b.safetensors")?,
        Model::SwinV2T => vs.load("safetensors/swin_v2_t.safetensors")?,
        Model::SwinV2S => vs.load("safetensors/swin_v2_s.safetensors")?,
        Model::SwinV2B => vs.load("safetensors/swin_v2_b.safetensors")?,
    }

    let transforms = match model {
        Model::AlexNet => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::ConvNextTiny => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            236,
            FilterType::Triangle,
        ),
        Model::ConvNextSmall => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            230,
            FilterType::Triangle,
        ),
        Model::ConvNextBase => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::ConvNextLarge => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::DenseNet121 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::DenseNet161 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::DenseNet169 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::DenseNet201 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::EfficientNetB0 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::CatmullRom,
        ),
        Model::EfficientNetB1 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            240,
            255,
            FilterType::Triangle,
        ),
        Model::EfficientNetB2 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            288,
            288,
            FilterType::CatmullRom,
        ),
        Model::EfficientNetB3 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            300,
            320,
            FilterType::CatmullRom,
        ),
        Model::EfficientNetB4 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            380,
            384,
            FilterType::CatmullRom,
        ),
        Model::EfficientNetB5 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            456,
            456,
            FilterType::CatmullRom,
        ),
        Model::EfficientNetB6 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            528,
            528,
            FilterType::CatmullRom,
        ),
        Model::EfficientNetB7 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            600,
            600,
            FilterType::CatmullRom,
        ),
        Model::EfficientNetV2S => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            384,
            384,
            FilterType::Triangle,
        ),
        Model::EfficientNetV2M => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            480,
            480,
            FilterType::Triangle,
        ),
        Model::EfficientNetV2L => Transforms::new(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
            480,
            480,
            FilterType::CatmullRom,
        ),
        Model::GoogleNet => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::InceptionV3 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            299,
            342,
            FilterType::Triangle,
        ),
        Model::MnasNet0_5 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::MnasNet0_75 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::MnasNet1_0 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::MnasNet1_3 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::MobileNetV2 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::MobileNetV3Large => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::MobileNetV3Small => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::RegNetY400MF => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::RegNetY800MF => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::RegNetY1_6GF => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::RegNetY3_2GF => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::RegNetY8GF => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::RegNetY16GF => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::RegNetY32GF => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::RegNetY128GF => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            384,
            384,
            FilterType::Triangle,
        ),
        Model::RegNetX400MF => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::RegNetX800MF => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::RegNetX1_6GF => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::RegNetX3_2GF => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::RegNetX8GF => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::RegNetX16GF => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::RegNetX32GF => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::ResNet18 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::ResNet34 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::ResNet50 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::ResNet101 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::ResNet152 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::ResNext50_32x4d => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::ResNext101_32x8d => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::ResNext101_64x4d => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::WideResNet50_2 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::WideResNet101_2 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::ShuffleNetV2X0_5 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::ShuffleNetV2X1_0 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::ShuffleNetV2X1_5 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::ShuffleNetV2X2_0 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::Triangle,
        ),
        Model::SqueezeNet1_0 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::SqueezeNet1_1 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::VGG11 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::VGG11BN => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::VGG13 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::VGG13BN => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::VGG16 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::VGG16BN => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::VGG19 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::VGG19BN => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::ViTB16 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::ViTB32 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::ViTL16 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            242,
            FilterType::Triangle,
        ),
        Model::ViTL32 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            256,
            FilterType::Triangle,
        ),
        Model::ViTBH14 => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            518,
            518,
            FilterType::CatmullRom,
        ),
        Model::SwinT => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            232,
            FilterType::CatmullRom,
        ),
        Model::SwinS => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            246,
            FilterType::CatmullRom,
        ),
        Model::SwinB => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            238,
            FilterType::CatmullRom,
        ),
        Model::SwinV2T => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            256,
            260,
            FilterType::CatmullRom,
        ),
        Model::SwinV2S => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            256,
            260,
            FilterType::CatmullRom,
        ),
        Model::SwinV2B => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            256,
            272,
            FilterType::CatmullRom,
        ),
        Model::MaxViTT => Transforms::new(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            224,
            224,
            FilterType::CatmullRom,
        )
    };

    Ok((m, vs, transforms))
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
        // // Model::RegNetY128GF,
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
        Model::EfficientNetB5,
        Model::EfficientNetB6,
        Model::EfficientNetB7,
        Model::EfficientNetV2S,
        Model::EfficientNetV2M,
        Model::EfficientNetV2L,
        Model::ViTB16,
        Model::ViTB32,
        // Model::ViTBH14,
        Model::ViTL16,
        Model::ViTL32,
        Model::MaxViTT,
        Model::SwinT,
        Model::SwinS,
        Model::SwinB,
        Model::SwinV2T,
        Model::SwinV2S,
        Model::SwinV2B,
    ] {
        println!("Loading model: {:?}", model);

        // eval_imagenet(model.clone());
        let _guard = tch::no_grad_guard();

        let (model, vs, transforms) = load_model(model)?;
        // let image = load_image_and_resize224("tiger.jpg")?
        //     .to_device(vs.device())
        //     .unsqueeze(0);
        // // let image = Tensor::ones([1, 3, 224, 224], (Kind::Float, vs.device()));

        // // let w = &vs.variables_.lock().unwrap().named_variables;
        // // let w = w.get("features.6.1.block.3.1.running_mean").unwrap();
        // // println!("{w}");

        // let output = model.forward_t(&image, false).softmax(-1, Kind::Float);

        // for (probability, class) in tch::vision::imagenet::top(&output, 5).iter() {
        //     println!("{:50} {:5.2}%", class, 100.0 * probability)
        // }
        // println!();

        // print_varstore(vs);

        let ds = imagenet::ImageNetDataset::new(PathBuf::from("."), transforms);
        let loader = DataLoader::builder(ds)
            .batch_size(64)
            .collate_fn(TorchCollate)
            .build();

        let mut y_true: Vec<Tensor> = vec![];
        let mut y_pred: Vec<Tensor> = vec![];

        for (_batch_id, (image, labels)) in loader.iter().enumerate().progress() {
            let output = model
                .forward_t(&image.to(vs.device()), false)
                .softmax(-1, Kind::Float)
                .argmax(1, false);
            y_true.push(labels.to(vs.device()));
            y_pred.push(output);
        }
        let y_true = Tensor::cat(&y_true, 0);
        let y_pred = Tensor::cat(&y_pred, 0);
        let accuracy = y_true
            .eq_tensor(&y_pred)
            .to_kind(Kind::Float)
            .mean(Kind::Float);
        println!("Accuracy: {:?}", accuracy);
    }

    Ok(())
}
