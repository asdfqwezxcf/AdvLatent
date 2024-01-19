import torch
from torch import nn
from collections import OrderedDict

class LatentWrapper(nn.Module):
    def __init__(self,model) -> None:
        super().__init__()
        feature = OrderedDict()
        tail = OrderedDict()
        tail_dict = ['layer4','avgpool','flatten','fc']
        for child_name, child_module in model.named_children():
            if child_name not in tail_dict:
                feature[child_name] = child_module
            if child_name in tail_dict:
                if child_name == 'fc':
                    tail['flatten'] = nn.Flatten(1)
                tail[child_name] = child_module
        self.feature = nn.Sequential(feature)
        self.tail = nn.Sequential(tail)
    def forward(self,x):
        latent = self.feature(x)
        output = self.tail(latent)
        return latent, output    

class AdvWrapper(nn.Module):
    def __init__(self,model,mean=(0.485,0.456,0.406),std=(0.229, 0.224, 0.225)) -> None:
        super().__init__()
        self.model = model
        self.mean = torch.tensor(mean).view(3,1,1)
        self.std = torch.tensor(std).view(3,1,1)

    def forward(self,x):
        device = x.device
        x = (x - self.mean.to(device)) / self.std.to(device)
        return self.model(x)

class SplitResNetHead(nn.Module):
    def __init__(self,resnet) -> None:
        super().__init__()
        model = resnet
        sequential = OrderedDict()
        tail_dict = ['layer3','layer4','avgpool','flatten','fc']
        for child_name, child_module in model.named_children():
            if child_name not in tail_dict:
                sequential[child_name] = child_module
        self.head = nn.Sequential(sequential)
    def forward(self,x):
        return self.head(x)

class SplitResNetTail(nn.Module):
    def __init__(self,resnet) -> None:
        super().__init__()
        model = resnet
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