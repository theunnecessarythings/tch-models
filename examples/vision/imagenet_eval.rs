use std::path::PathBuf;

use ai_dataloader::collate::TorchCollate;
use ai_dataloader::indexable::DataLoader;
use anyhow::Result;
use imagenet::Transforms;
use indicatif::ProgressIterator;
use tch::{
    nn::{self, VarStore},
    vision::imagenet::load_image_and_resize224,
    Kind, Tensor,
};
use tch_models::vision::{imagenet, models::Model};

fn eval_imagenet(vs: &VarStore, model: Box<dyn nn::ModuleT>, transforms: Transforms) {
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

fn test_torchvision_models() -> Result<()> {
    for model in [
        Model::AlexNet,
        Model::ConvNextBase,
        Model::ConvNextSmall,
        Model::ConvNextTiny,
        Model::ConvNextLarge,
        Model::DenseNet121,
        Model::DenseNet161,
        Model::DenseNet169,
        Model::DenseNet201,
        Model::GoogleNet,
        Model::InceptionV3,
        Model::MnasNet0_5,
        Model::MnasNet0_75,
        Model::MnasNet1_0,
        Model::MnasNet1_3,
        Model::MobileNetV2,
        Model::MobileNetV3Small,
        Model::MobileNetV3Large,
        Model::RegNetY400MF,
        Model::RegNetY800MF,
        Model::RegNetY1_6GF,
        Model::RegNetY3_2GF,
        Model::RegNetY8GF,
        Model::RegNetY16GF,
        Model::RegNetY32GF,
        Model::RegNetX400MF,
        Model::RegNetX800MF,
        Model::RegNetX1_6GF,
        Model::RegNetX3_2GF,
        Model::RegNetX8GF,
        Model::RegNetX16GF,
        Model::RegNetX32GF,
        Model::ShuffleNetV2X0_5,
        Model::ShuffleNetV2X1_0,
        Model::ShuffleNetV2X1_5,
        Model::ShuffleNetV2X2_0,
        Model::SqueezeNet1_0,
        Model::SqueezeNet1_1,
        Model::VGG11,
        Model::VGG11BN,
        Model::VGG13,
        Model::VGG13BN,
        Model::VGG16,
        Model::VGG16BN,
        Model::VGG19,
        Model::VGG19BN,
        Model::ResNet18,
        Model::ResNet34,
        Model::ResNet50,
        Model::ResNet101,
        Model::ResNet152,
        Model::ResNext50_32x4d,
        Model::ResNext101_32x8d,
        Model::ResNext101_64x4d,
        Model::WideResNet50_2,
        Model::WideResNet101_2,
        Model::EfficientNetB0,
        Model::EfficientNetB1,
        Model::EfficientNetB2,
        Model::EfficientNetB3,
        Model::EfficientNetB4,
        Model::EfficientNetB5,
        Model::EfficientNetB6,
        Model::EfficientNetB7,
        Model::EfficientNetV2S,
        Model::EfficientNetV2M,
        Model::EfficientNetV2L,
        Model::ViTB16,
        Model::ViTB32,
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

        let _guard = tch::no_grad_guard();

        // Test with a single image
        let (model, vs, transforms) = tch_models::vision::models::load_model(model, true)?;
        let image = load_image_and_resize224("examples/vision/tiger.jpg")?
            .to_device(vs.device())
            .unsqueeze(0);

        /*
        let image = Tensor::ones([1, 3, 224, 224], (Kind::Float, vs.device()));

        let w = &vs.variables_.lock().unwrap().named_variables;
        let w = w.get("features.6.1.block.3.1.running_mean").unwrap();
        println!("{w}");
        */

        let output = model.forward_t(&image, false).softmax(-1, Kind::Float);

        for (probability, class) in tch::vision::imagenet::top(&output, 5).iter() {
            println!("{:50} {:5.2}%", class, 100.0 * probability)
        }
        println!();

        // tch_models::vision::models::print_varstore(&vs);

        eval_imagenet(&vs, model, transforms);
    }
    Ok(())
}

fn main() -> Result<()> {
    test_torchvision_models()?;
    Ok(())
}
