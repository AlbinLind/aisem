Trained model could possibly be improved further by training more.

Model
```python
class CNN(nn.Module):
    def __init__(self, output_dim: int):
        super(CNN, self).__init__()
        self.output_dim = output_dim
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.requires_grad_(False)
        self.sequential = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )
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
    "0801-0850": 0,
    "0851-0900": 0,
    "0751-0800": 0,
    "0951-1000": 0,
    "1001-1050": 0,
    "1051-1100": 0,
    "1101-1150": 0,
    "1151-1200": 0,
    "1201-1250": 0,
    "1251-1300": 0,
    "1301-1350": 0,
    "1351-1400": 0,
    "1401-1450": 1,
    "1451-1500": 1,
    "1501-1550": 2,
    "1551-1600": 2,
    "1601-1650": 3,
    "1651-1700": 3,
    "1701-1750": 4,
    "1751-1800": 4,
    "1801-1850": 5,
    "1851-1900": 5,
}
```