### Sanity check using Torchvision weights on ImageNet (Top-1 Accuracy)

| Model                | tch-rs             | TorchVision                |
|----------------------|--------------------|----------------------------|
| AlexNet              | 23.33              | 56.52                      |
| ConvNextBase         | 84.04              | 84.06                      |
| ConvNextSmall        | 83.56              | 83.62                      |
| ConvNextTiny         | 82.47              | 82.52                      |
| ConvNextLarge        | 84.39              | 84.41                      |
| DenseNet121          | 74.39              | 74.43                      |
| DenseNet161          | 77.16              | 77.14                      |
| DenseNet169          | 75.56              | 75.60                      |
| DenseNet201          | 76.89              | 76.90                      |
| GoogleNet            | 69.75              | 69.78                      |
| InceptionV3          | 77.18              | 77.30                      |
| MnasNet0_5           | 67.71              | 67.73                      |
| MnasNet0_75          | 71.16              | 71.18                      |
| MnasNet1_0           | 73.42              | 73.46                      |
| MnasNet1_3           | 76.47              | 76.51                      |
| MobileNetV2          | 71.70              | 71.88                      |
| MobileNetV3Small     | 43.34              | 67.67                      |
| MobileNetV3Large     | 62.56              | 74.04                      |
| RegNetY400MF         | 73.79              | 74.05                      |
| RegNetY800MF         | 76.42              | 76.42                      |
| RegNetY1_6GF         | 77.69              | 77.95                      |
| RegNetY3_2GF         | 78.81              | 78.95                      |
| RegNetY8GF           | 79.99              | 80.03                      |
| RegNetY16GF          | 80.53              | 80.42                      |
| RegNetY32GF          | 80.91              | 80.88                      |
| RegNetX400MF         | 72.49              | 72.83                      |
| RegNetX800MF         | 74.86              | 75.21                      |
| RegNetX1_6GF         | 77.07              | 77.04                      |
| RegNetX3_2GF         | 78.40              | 78.36                      |
| RegNetX8GF           | 79.45              | 79.34                      |
| RegNetX16GF          | 80.11              | 80.06                      |
| RegNetX32GF          | 80.47              | 80.62                      |
| ShuffleNetV2X0_5     | 60.49              | 60.55                      |
| ShuffleNetV2X1_0     | 69.25              | 69.36                      |
| ShuffleNetV2X1_5     | 73.07              | 73.00                      |
| ShuffleNetV2X2_0     | 76.25              | 76.23                      |
| SqueezeNet1_0        | 56.98              | 58.09                      |
| SqueezeNet1_1        | 58.19              | 58.18                      |
| VGG11                | 68.95              | 69.02                      |
| VGG11BN              | 70.34              | 70.37                      |
| VGG13                | 69.89              | 69.93                      |
| VGG13BN              | 71.59              | 71.56                      |
| VGG16                | 71.62              | 71.60                      |
| VGG16BN              | 73.46              | 73.36                      |
| VGG19                | 72.37              | 72.38                      |
| VGG19BN              | 74.14              | 74.22                      |
| ResNet18             | 69.79              | 69.76                      |
| ResNet34             | 73.30              | 73.31                      |
| ResNet50             | 75.90              | 76.13                      |
| ResNet101            | 77.25              | 77.37                      |
| ResNet152            | 78.26              | 78.31                      |
| ResNext50_32x4d      | 77.61              | 77.62                      |
| ResNext101_32x8d     | 79.01              | 79.31                      |
| ResNext101_64x4d     | 83.23              | 83.25                      |
| WideResNet50_2       | 78.46              | 78.47                      |
| WideResNet101_2      | 78.89              | 78.85                      |
| EfficientNetB0       | 73.92              | 77.70                      |
| EfficientNetB1       | 77.89              | 78.64                      |
| EfficientNetB2       | 77.54              | 80.61                      |
| EfficientNetB3       | 81.30              | 82.00                      |
| EfficientNetB4       | 83.37              | 83.38                      |
| EfficientNetB5       | 83.49              | 83.44                      |
| EfficientNetB6       | 84.00              | 84.00                      |
| EfficientNetB7       | 84.12              | 84.12                      |
| EfficientNetV2S      | 84.17              | 84.23                      |
| EfficientNetV2M      | 85.12              | 85.11                      |
| EfficientNetV2L      | 85.81              | 85.81                      |
| ViTB16               | 81.05              | 81.07                      |
| ViTB32               | 75.88              | 75.91                      |
| ViTL16               | 79.61              | 79.66                      |
| ViTL32               | 76.98              | 76.97                      |
| MaxViTT              | 83.69              | 83.70                      |
| SwinT                | 81.40              | 81.47                      |
| SwinS                | 83.16              | 83.20                      |
| SwinB                | 83.61              | 83.58                      |
| SwinV2T              | 82.03              | 82.07                      |
| SwinV2S              | 83.69              | 83.71                      |
| SwinV2B              | 84.08              | 84.11                      |


## TODO
- [x] ConvNet Model Implementations
- [x] Weight Loading from PyTorch
- [ ] Proper weight initializations
- [x] ViT Models
- [x] ImageNet Evaluations
- [ ] CIFAR-10 Training 
- [ ] Retain original comments
- [ ] Add references
- [ ] Documentation
- [ ] Fix Bugs 
    - [ ] AlexNet
    - [ ] MobileNetV3
