import torch
from torch import nn

class Inception_3a(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.model_2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=96, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.model_3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.model_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=192, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,x):
        return torch.cat([
            self.model_1(x),
            self.model_2(x),
            self.model_3(x),
            self.model_4(x)
        ], dim=1)

class Inception_3b(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.model_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.model_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.model_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,x):
        return torch.cat([
            self.model_1(x),
            self.model_2(x),
            self.model_3(x),
            self.model_4(x)
        ], dim=1)

class Inception_4a(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_1 = nn.Sequential(
            nn.Conv2d(in_channels=480, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.model_2 = nn.Sequential(
            nn.Conv2d(in_channels=480, out_channels=96, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=208, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.model_3 = nn.Sequential(
            nn.Conv2d(in_channels=480, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.model_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=480, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,x):
        return torch.cat([
            self.model_1(x),
            self.model_2(x),
            self.model_3(x),
            self.model_4(x)
        ], dim=1)

class Inception_4b(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=160, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.model_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=112, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=112, out_channels=224, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.model_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=24, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.model_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,x):
        return torch.cat([
            self.model_1(x),
            self.model_2(x),
            self.model_3(x),
            self.model_4(x)
        ], dim=1)

class Inception_4c(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.model_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.model_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=24, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.model_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,x):
        return torch.cat([
            self.model_1(x),
            self.model_2(x),
            self.model_3(x),
            self.model_4(x)
        ], dim=1)

class Inception_4d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=112, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.model_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=144, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=144, out_channels=288, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.model_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.model_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,x):
        return torch.cat([
            self.model_1(x),
            self.model_2(x),
            self.model_3(x),
            self.model_4(x)
        ], dim=1)

class Inception_4e(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_1 = nn.Sequential(
            nn.Conv2d(in_channels=528, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.model_2 = nn.Sequential(
            nn.Conv2d(in_channels=528, out_channels=160, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=160, out_channels=320, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.model_3 = nn.Sequential(
            nn.Conv2d(in_channels=528, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.model_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=528, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,x):
        return torch.cat([
            self.model_1(x),
            self.model_2(x),
            self.model_3(x),
            self.model_4(x)
        ], dim=1)

class Inception_5a(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_1 = nn.Sequential(
            nn.Conv2d(in_channels=832, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.model_2 = nn.Sequential(
            nn.Conv2d(in_channels=832, out_channels=160, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=160, out_channels=320, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.model_3 = nn.Sequential(
            nn.Conv2d(in_channels=832, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.model_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=832, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,x):
        return torch.cat([
            self.model_1(x),
            self.model_2(x),
            self.model_3(x),
            self.model_4(x)
        ], dim=1)

class Inception_5b(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_1 = nn.Sequential(
            nn.Conv2d(in_channels=832, out_channels=384, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.model_2 = nn.Sequential(
            nn.Conv2d(in_channels=832, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.model_3 = nn.Sequential(
            nn.Conv2d(in_channels=832, out_channels=48, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.model_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=832, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,x):
        return torch.cat([
            self.model_1(x),
            self.model_2(x),
            self.model_3(x),
            self.model_4(x)
        ], dim=1)

class AuxiliaryClassificationNetwork(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3, padding=0),
            nn.Conv2d(in_channels=in_channel, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=out_channel)
        )

    def forward(self, x):
        return self.model(x)

class GoogleNet(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.preprocessing = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.inception_3a = Inception_3a()
        self.inception_3b = Inception_3b()
        self.mp_3b24a = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception_4a = Inception_4a()
        self.inception_4b = Inception_4b()
        self.inception_4c = Inception_4c()
        self.inception_4d = Inception_4d()
        self.inception_4e = Inception_4e()
        self.mp_4e25a = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception_5a = Inception_5a()
        self.inception_5b = Inception_5b()
        
        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=1024, out_features=out_channel)
        )

        self.auxiliary_classifier_4a = AuxiliaryClassificationNetwork(in_channel=512, out_channel=out_channel)
        self.auxiliary_classifier_4d = AuxiliaryClassificationNetwork(in_channel=528, out_channel=out_channel)

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):                            # 若是卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out',   # 用（何）kaiming_normal_法初始化权重
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)                    # 初始化偏重为0
            elif isinstance(m, nn.Linear):            # 若是全连接层
                nn.init.normal_(m.weight, 0, 0.01)    # 正态分布初始化
                nn.init.constant_(m.bias, 0)          # 初始化偏重为0

    def forward(self, x):
        tool = self.preprocessing(x)
        tool = self.inception_3a(tool)
        tool = self.inception_3b(tool)
        tool = self.mp_3b24a(tool)
        tool = self.inception_4a(tool)
        aux_result_1 = self.auxiliary_classifier_4a(tool)
        tool = self.inception_4b(tool)
        tool = self.inception_4c(tool)
        tool = self.inception_4d(tool)
        aux_result_2 = self.auxiliary_classifier_4d(tool)
        tool = self.inception_4e(tool)
        tool = self.mp_4e25a(tool)
        tool = self.inception_5a(tool)
        tool = self.inception_5b(tool)
        result = self.classifier(tool)

        return result, aux_result_1, aux_result_2
        
