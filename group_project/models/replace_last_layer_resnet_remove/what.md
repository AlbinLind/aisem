Model
```python
class CNN(nn.Module):
    def __init__(self, output_dim: int):
        super(CNN, self).__init__()
        self.output_dim = output_dim
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.requires_grad_(False)
        self.resnet.fc = nn.Linear(2048, output_dim)
        self.resnet.fc.requires_grad_(True)
```

Transform
```python
        self.transform = transforms.Compose(
            [
                # Get the values from here: https://pytorch.org/vision/0.18/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
                transforms.Resize(232),
                # Data augmentation, randomly crop the image and flip it horizontally
                # only for the training set
                transforms.RandomCrop(224)
                if data_set == DataSetEnum.TRAIN
                else transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip()
                if data_set == DataSetEnum.TRAIN
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
```

Encoding
```python
timeframe_encoder = {
    "1401-1450": 0,
    "1451-1500": 1,
    "1501-1550": 2,
    "1551-1600": 3,
    "1601-1650": 4,
    "1651-1700": 5,
    "1701-1750": 6,
    "1751-1800": 7,
    "1801-1850": 8,
    "1851-1900": 9,
}
```