### Sanity check using Torchvision weights on ImageNet (Top-1 Accuracy)

| Model                | tch-rs             | TorchVision           |
|----------------------|--------------------|-----------------------|
| AlexNet              | 23.33              |                       |
| ConvNextBase         | 84.04              |                       |
| ConvNextSmall        | 83.56              |                       |
| ConvNextTiny         | 82.47              |                       |
| ConvNextLarge        | 84.39              |                       |
| DenseNet121          | 74.39              |                       |
| DenseNet161          | 77.16              |                       |
| DenseNet169          | 75.56              |                       |
| DenseNet201          | 76.89              |                       |
| GoogleNet            | 69.75              |                       |
| InceptionV3          | 77.18              |                       |
| MnasNet0_5           | 67.71              |                       |
| MnasNet0_75          | 71.16              |                       |
| MnasNet1_0           | 73.42              |                       |
| MnasNet1_3           | 76.47              |                       |
| MobileNetV2          | 71.70              |                       |
| MobileNetV3Small     | 43.34              |                       |
| MobileNetV3Large     | 62.56              |                       |
| RegNetY400MF         | 73.79              |                       |
| RegNetY800MF         | 76.42              |                       |
| RegNetY1_6GF         | 77.69              |                       |
| RegNetY3_2GF         | 78.81              |                       |
| RegNetY8GF           | 79.99              |                       |
| RegNetY16GF          | 80.53              |                       |
| RegNetY32GF          | 80.91              |                       |
| RegNetX400MF         | 72.49              |                       |
| RegNetX800MF         | 74.86              |                       |
| RegNetX1_6GF         | 77.07              |                       |
| RegNetX3_2GF         | 78.40              |                       |
| RegNetX8GF           | 79.45              |                       |
| RegNetX16GF          | 80.11              |                       |
| RegNetX32GF          | 80.47              |                       |
| ShuffleNetV2X0_5     | 60.49              |                       |
| ShuffleNetV2X1_0     | 69.25              |                       |
| ShuffleNetV2X1_5     | 73.07              |                       |
| ShuffleNetV2X2_0     | 76.25              |                       |
| SqueezeNet1_0        | 56.98              |                       |
| SqueezeNet1_1        | 58.19              |                       |
| VGG11                | 68.95              |                       |
| VGG11BN              | 70.34              |                       |
| VGG13                | 69.89              |                       |
| VGG13BN              | 71.59              |                       |
| VGG16                | 71.62              |                       |
| VGG16BN              | 73.46              |                       |
| VGG19                | 72.37              |                       |
| VGG19BN              | 74.14              |                       |
| ResNet18             | 69.79              |                       |
| ResNet34             | 73.30              |                       |
| ResNet50             | 75.90              |                       |
| ResNet101            | 77.25              |                       |
| ResNet152            | 78.26              |                       |
| ResNext50_32x4d      | 77.61              |                       |
| ResNext101_32x8d     | 79.01              |                       |
| ResNext101_64x4d     | 83.23              |                       |
| WideResNet50_2       | 78.46              |                       |
| WideResNet101_2      | 78.89              |                       |
| EfficientNetB0       | 73.92              |                       |
| EfficientNetB1       | 77.89              |                       |
| EfficientNetB2       | 77.54              |                       |
| EfficientNetB3       | 81.30              |                       |
| EfficientNetB4       | 83.37              |                       |
| EfficientNetB5       | 83.49              |                       |
| EfficientNetB6       | 84.00              |                       |
| EfficientNetB7       | 84.12              |                       |
| EfficientNetV2S      | 84.17              |                       |
| EfficientNetV2M      | 85.12              |                       |
| EfficientNetV2L      | 85.81              |                       |
| ViTB16               | 81.05              |                       |
| ViTB32               | 75.88              |                       |
| ViTL16               | 79.61              |                       |
| ViTL32               | 76.98              |                       |
| MaxViTT              | 83.69              |                       |
| SwinT                | 81.40              |                       |
| SwinS                | 83.16              |                       |
| SwinB                | 83.61              |                       |
| SwinV2T              | 82.03              |                       |
| SwinV2S              | 83.69              |                       |
| SwinV2B              | 84.08              |                       |


## TODO
- [x] ConvNet Model Implementations
- [x] Weight Loading from PyTorch
- [ ] Proper weight initializations
- [x] ViT Models
- [x] ImageNet Evaluations
- [ ] CIFAR-10 Training 
- [ ] Retain original comments
