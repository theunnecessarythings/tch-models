use image::imageops::FilterType;
use tch::{
    nn::{self, VarStore},
    Device,
};

use crate::vision::{
    alexnet, convnext, densenet, efficientnet, googlenet, imagenet::Transforms, inception, maxvit,
    mnasnet, mobilenet_v2, mobilenet_v3, regnet, resnet, shufflenet_v2, squeezenet,
    swin_transformer, vgg, vision_transformer,
};
use anyhow::Result;

#[derive(Debug, Clone)]
pub enum Model {
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

pub fn load_model(
    model: Model,
    pretrained: bool,
) -> Result<(Box<dyn nn::ModuleT>, VarStore, Transforms)> {
    let api = hf_hub::api::sync::Api::new().unwrap();
    let api = api.model("theunnecessarythings/vision_models".into());

    let mut vs = VarStore::new(Device::cuda_if_available());
    match model {
        Model::AlexNet => {
            let m = Box::new(alexnet::alexnet(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("alexnet.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ConvNextTiny => {
            let m = Box::new(convnext::convnext_tiny(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("convnext_tiny.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                236,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ConvNextSmall => {
            let m = Box::new(convnext::convnext_small(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("convnext_small.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                230,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ConvNextBase => {
            let m = Box::new(convnext::convnext_base(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("convnext_base.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ConvNextLarge => {
            let m = Box::new(convnext::convnext_large(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("convnext_large.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::DenseNet121 => {
            let m = Box::new(densenet::densenet121(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("densenet121.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::DenseNet161 => {
            let m = Box::new(densenet::densenet161(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("densenet161.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::DenseNet169 => {
            let m = Box::new(densenet::densenet169(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("densenet169.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::DenseNet201 => {
            let m = Box::new(densenet::densenet201(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("densenet201.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::EfficientNetB0 => {
            let m = Box::new(efficientnet::efficientnet_b0(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("efficientnet_b0.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::CatmullRom,
            );
            Ok((m, vs, t))
        }
        Model::EfficientNetB1 => {
            let m = Box::new(efficientnet::efficientnet_b1(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("efficientnet_b1.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                240,
                255,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::EfficientNetB2 => {
            let m = Box::new(efficientnet::efficientnet_b2(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("efficientnet_b2.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                288,
                288,
                FilterType::CatmullRom,
            );
            Ok((m, vs, t))
        }
        Model::EfficientNetB3 => {
            let m = Box::new(efficientnet::efficientnet_b3(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("efficientnet_b3.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                300,
                320,
                FilterType::CatmullRom,
            );
            Ok((m, vs, t))
        }
        Model::EfficientNetB4 => {
            let m = Box::new(efficientnet::efficientnet_b4(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("efficientnet_b4.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                380,
                384,
                FilterType::CatmullRom,
            );
            Ok((m, vs, t))
        }
        Model::EfficientNetB5 => {
            let m = Box::new(efficientnet::efficientnet_b5(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("efficientnet_b5.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                456,
                456,
                FilterType::CatmullRom,
            );
            Ok((m, vs, t))
        }
        Model::EfficientNetB6 => {
            let m = Box::new(efficientnet::efficientnet_b6(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("efficientnet_b6.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                528,
                528,
                FilterType::CatmullRom,
            );
            Ok((m, vs, t))
        }
        Model::EfficientNetB7 => {
            let m = Box::new(efficientnet::efficientnet_b7(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("efficientnet_b7.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                600,
                600,
                FilterType::CatmullRom,
            );
            Ok((m, vs, t))
        }
        Model::EfficientNetV2S => {
            let m = Box::new(efficientnet::efficientnet_v2s(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("efficientnet_v2_s.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                384,
                384,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::EfficientNetV2M => {
            let m = Box::new(efficientnet::efficientnet_v2m(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("efficientnet_v2_m.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                480,
                480,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::EfficientNetV2L => {
            let m = Box::new(efficientnet::efficientnet_v2l(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("efficientnet_v2_l.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
                480,
                480,
                FilterType::CatmullRom,
            );
            Ok((m, vs, t))
        }
        Model::GoogleNet => {
            let m = Box::new(googlenet::googlenet(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("googlenet.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::InceptionV3 => {
            let m = Box::new(inception::inception_v3(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("inception_v3.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                299,
                342,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::MnasNet0_5 => {
            let m = Box::new(mnasnet::mnasnet0_5(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("mnasnet0_5.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::MnasNet0_75 => {
            let m = Box::new(mnasnet::mnasnet0_75(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("mnasnet0_75.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::MnasNet1_0 => {
            let m = Box::new(mnasnet::mnasnet1_0(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("mnasnet1_0.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::MnasNet1_3 => {
            let m = Box::new(mnasnet::mnasnet1_3(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("mnasnet1_3.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::MobileNetV2 => {
            let m = Box::new(mobilenet_v2::mobilenet_v2(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("mobilenet_v2.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::MobileNetV3Large => {
            let m = Box::new(mobilenet_v3::mobilenet_v3_large(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("mobilenet_v3_large.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::MobileNetV3Small => {
            let m = Box::new(mobilenet_v3::mobilenet_v3_small(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("mobilenet_v3_small.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::RegNetY400MF => {
            let m = Box::new(regnet::regnet_y_400mf(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("regnet_y_400mf.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::RegNetY800MF => {
            let m = Box::new(regnet::regnet_y_800mf(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("regnet_y_800mf.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::RegNetY1_6GF => {
            let m = Box::new(regnet::regnet_y_1_6gf(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("regnet_y_1_6gf.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::RegNetY3_2GF => {
            let m = Box::new(regnet::regnet_y_3_2gf(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("regnet_y_3_2gf.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::RegNetY8GF => {
            let m = Box::new(regnet::regnet_y_8gf(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("regnet_y_8gf.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::RegNetY16GF => {
            let m = Box::new(regnet::regnet_y_16gf(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("regnet_y_16gf.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::RegNetY32GF => {
            let m = Box::new(regnet::regnet_y_32gf(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("regnet_y_32gf.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::RegNetY128GF => {
            let m = Box::new(regnet::regnet_y_128gf(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("regnet_y_128gf.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                384,
                384,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::RegNetX400MF => {
            let m = Box::new(regnet::regnet_x_400mf(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("regnet_x_400mf.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::RegNetX800MF => {
            let m = Box::new(regnet::regnet_x_800mf(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("regnet_x_800mf.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::RegNetX1_6GF => {
            let m = Box::new(regnet::regnet_x_1_6gf(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("regnet_x_1_6gf.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::RegNetX3_2GF => {
            let m = Box::new(regnet::regnet_x_3_2gf(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("regnet_x_3_2gf.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::RegNetX8GF => {
            let m = Box::new(regnet::regnet_x_8gf(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("regnet_x_8gf.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::RegNetX16GF => {
            let m = Box::new(regnet::regnet_x_16gf(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("regnet_x_16gf.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::RegNetX32GF => {
            let m = Box::new(regnet::regnet_x_32gf(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("regnet_x_32gf.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ResNet18 => {
            let m = Box::new(resnet::resnet18(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("resnet18.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ResNet34 => {
            let m = Box::new(resnet::resnet34(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("resnet34.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ResNet50 => {
            let m = Box::new(resnet::resnet50(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("resnet50.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ResNet101 => {
            let m = Box::new(resnet::resnet101(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("resnet101.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ResNet152 => {
            let m = Box::new(resnet::resnet152(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("resnet152.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ResNext50_32x4d => {
            let m = Box::new(resnet::resnext50_32x4d(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("resnext50_32x4d.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ResNext101_32x8d => {
            let m = Box::new(resnet::resnext101_32x8d(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("resnext101_32x8d.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ResNext101_64x4d => {
            let m = Box::new(resnet::resnext101_64x4d(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("resnext101_64x4d.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::WideResNet50_2 => {
            let m = Box::new(resnet::wide_resnet50_2(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("wide_resnet50_2.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::WideResNet101_2 => {
            let m = Box::new(resnet::wide_resnet101_2(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("wide_resnet101_2.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ShuffleNetV2X0_5 => {
            let m = Box::new(shufflenet_v2::shufflenet_v2_x0_5(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("shufflenet_v2_x0_5.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ShuffleNetV2X1_0 => {
            let m = Box::new(shufflenet_v2::shufflenet_v2_x1_0(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("shufflenet_v2_x1_0.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ShuffleNetV2X1_5 => {
            let m = Box::new(shufflenet_v2::shufflenet_v2_x1_5(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("shufflenet_v2_x1_5.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ShuffleNetV2X2_0 => {
            let m = Box::new(shufflenet_v2::shufflenet_v2_x2_0(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("shufflenet_v2_x2_0.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::SqueezeNet1_0 => {
            let m = Box::new(squeezenet::squeezenet1_0(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("squeezenet1_0.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::SqueezeNet1_1 => {
            let m = Box::new(squeezenet::squeezenet1_1(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("squeezenet1_1.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::VGG11 => {
            let m = Box::new(vgg::vgg11(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("vgg11.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::VGG11BN => {
            let m = Box::new(vgg::vgg11_bn(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("vgg11_bn.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::VGG13 => {
            let m = Box::new(vgg::vgg13(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("vgg13.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::VGG13BN => {
            let m = Box::new(vgg::vgg13_bn(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("vgg13_bn.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::VGG16 => {
            let m = Box::new(vgg::vgg16(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("vgg16.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::VGG16BN => {
            let m = Box::new(vgg::vgg16_bn(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("vgg16_bn.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::VGG19 => {
            let m = Box::new(vgg::vgg19(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("vgg19.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::VGG19BN => {
            let m = Box::new(vgg::vgg19_bn(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("vgg19_bn.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ViTB16 => {
            let m = Box::new(vision_transformer::vit_b_16(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("vit_b_16.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ViTB32 => {
            let m = Box::new(vision_transformer::vit_b_32(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("vit_b_32.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ViTL16 => {
            let m = Box::new(vision_transformer::vit_l_16(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("vit_l_16.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                242,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ViTL32 => {
            let m = Box::new(vision_transformer::vit_l_32(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("vit_l_32.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                256,
                FilterType::Triangle,
            );
            Ok((m, vs, t))
        }
        Model::ViTBH14 => {
            let m = Box::new(vision_transformer::vit_h_14(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("vit_h_14.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                518,
                518,
                FilterType::CatmullRom,
            );
            Ok((m, vs, t))
        }
        Model::SwinT => {
            let m = Box::new(swin_transformer::swin_t(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("swin_t.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                232,
                FilterType::CatmullRom,
            );
            Ok((m, vs, t))
        }
        Model::SwinS => {
            let m = Box::new(swin_transformer::swin_s(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("swin_s.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                246,
                FilterType::CatmullRom,
            );
            Ok((m, vs, t))
        }
        Model::SwinB => {
            let m = Box::new(swin_transformer::swin_b(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("swin_b.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                238,
                FilterType::CatmullRom,
            );
            Ok((m, vs, t))
        }
        Model::SwinV2T => {
            let m = Box::new(swin_transformer::swin_v2_t(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("swin_v2_t.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                256,
                260,
                FilterType::CatmullRom,
            );
            Ok((m, vs, t))
        }
        Model::SwinV2S => {
            let m = Box::new(swin_transformer::swin_v2_s(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("swin_v2_s.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                256,
                260,
                FilterType::CatmullRom,
            );
            Ok((m, vs, t))
        }
        Model::SwinV2B => {
            let m = Box::new(swin_transformer::swin_v2_b(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("swin_v2_b.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                256,
                272,
                FilterType::CatmullRom,
            );
            Ok((m, vs, t))
        }
        Model::MaxViTT => {
            let m = Box::new(maxvit::maxvit_t(&vs.root(), 1000));
            if pretrained {
                vs.load(api.get("maxvit_t.safetensors")?)?;
            }
            let t = Transforms::new(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                224,
                224,
                FilterType::CatmullRom,
            );
            Ok((m, vs, t))
        }
    }
}

pub fn print_varstore(vs: &VarStore) {
    let named_variables: Vec<(String, Vec<i64>)> = vs
        .variables_
        .lock()
        .unwrap()
        .named_variables
        .iter()
        .map(|(key, val)| (key.clone(), val.size()))
        .collect();

    for (key, shape) in named_variables {
        println!("{:?} {:?}", key, shape);
    }
}
