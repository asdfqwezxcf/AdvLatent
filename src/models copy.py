import torch
from torch import nn
from collections import OrderedDict
from torchvision.models import resnet152

class Bottleneck(nn.Module):
    def __init__(self, bottleneck_channel) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False)
        )

        self.decoder = nn.Sequential(
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, 512, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
    def forward(self,x):
        encode_feature = self.encoder(x)
        decode_feature = self.decoder(encode_feature)
        return encode_feature, decode_feature
    
class CustomResNet152(nn.Module):
    def __init__(self,bottleneck_channel=12) -> None:
        super().__init__()
        self.bottleneck = Bottleneck(bottleneck_channel)
        resnet152model = resnet152(weights=None)
        self.layer3 = resnet152model.layer3
        self.layer4 = resnet152model.layer4
        self.avgpool = resnet152model.avgpool
        self.fc = resnet152model.fc

    def forward(self,x):
        encode_feature,decode_feature = self.bottleneck(x)
        output = self.layer3(decode_feature)
        output = self.layer4(output)
        output = self.avgpool(output)
        output = torch.squeeze(output)
        output = self.fc(output)
        return output
    
class AdvModel(nn.Module):
    def __init__(self,model,mean=(0.485,0.456,0.406),std=(0.229, 0.224, 0.225)) -> None:
        super().__init__()
        self.model = model
        self.mean = torch.tensor(mean).view(3,1,1)
        self.std = torch.tensor(std).view(3,1,1)

    def forward(self,x):
        device = x.get_device()
        if device >= 0:
            self.mean = self.mean.to('cuda:%d'%device)
            self.std = self.std.to('cuda:%d'%device)
        x = (x - self.mean) / self.std
        return self.model(x)
    
class SplitResNetHead(nn.Module):
    def __init__(self,weight="IMAGENET1K_V1",mean=(0.485,0.456,0.406),std=(0.229, 0.224, 0.225)) -> None:
        super().__init__()
        model = resnet152(weights=weight)
        sequential = OrderedDict()
        tail_dict = ['layer3','layer4','avgpool','fc']
        for child_name, child_module in model.named_children():
            if child_name not in tail_dict:
                sequential[child_name] = child_module
        self.head = nn.Sequential(sequential)
        self.mean = torch.tensor(mean).view(3,1,1)
        self.std = torch.tensor(std).view(3,1,1)
    def forward(self,x):
        device = x.get_device()
        if device >= 0:
            self.mean = self.mean.to('cuda:%d'%device)
            self.std = self.std.to('cuda:%d'%device)
        x = (x - self.mean) / self.std
        return self.head(x)

class SplitResNetTail(nn.Module):
    def __init__(self,weight="IMAGENET1K_V1") -> None:
        super().__init__()
        model = resnet152(weights=weight)
        sequential = OrderedDict()
        tail_dict = ['layer3','layer4','avgpool','fc']
        for child_name, child_module in model.named_children():
            if child_name in tail_dict:
                if child_name == 'fc':
                    sequential['flatten'] = nn.Flatten(1)
                sequential[child_name] = child_module
        self.tail = nn.Sequential(sequential)
        
    def forward(self,x):
        return self.tail(x)

class SplitHead(nn.Module):
    def __init__(self,pretrained_model,mean=(0.485,0.456,0.406),std=(0.229, 0.224, 0.225)) -> None:
        super().__init__()
        self.mean = torch.tensor(mean).view(3,1,1)
        self.std = torch.tensor(std).view(3,1,1)
        self.head = pretrained_model.bottleneck.encoder
    def forward(self,x):
        device = x.get_device()
        if device >= 0:
            self.mean = self.mean.to('cuda:%d'%device)
            self.std = self.std.to('cuda:%d'%device)
        x = (x - self.mean) / self.std
        return self.head(x)
    
class SplitTail(nn.Module):
    def __init__(self,pretrianed_model) -> None:
        super().__init__()
        self.decoder = pretrianed_model.bottleneck.decoder
        self.layer3 = pretrianed_model.layer3
        self.layer4 = pretrianed_model.layer4
        self.avgpool = pretrianed_model.avgpool
        self.fc = pretrianed_model.fc

    def forward(self,x):
        decode_feature = self.decoder(x)
        output = self.layer3(decode_feature)
        output = self.layer4(output)
        output = self.avgpool(output)
        output = torch.squeeze(output)
        output = self.fc(output)
        return output