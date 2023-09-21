import torch
from torch import nn
from collections import OrderedDict
from torchvision.models import resnet152, vgg16_bn

class Bottleneck(nn.Module):
    def __init__(self, bottleneck_channel, backbone='resnet') -> None:
        super().__init__()
        if backbone == 'resnet':
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
        elif backbone == 'vgg':
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(64, bottleneck_channel, kernel_size=3, stride=1, padding=1)
            )

            self.decoder = nn.Sequential(
                nn.BatchNorm2d(bottleneck_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(bottleneck_channel, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )

    def forward(self,x):
        encode_feature = self.encoder(x)
        decode_feature = self.decoder(encode_feature)
        return decode_feature
    
class CustomResNet152(nn.Module):
    def __init__(self,bottleneck_channel=12) -> None:
        super().__init__()
        self.bottleneck = Bottleneck(bottleneck_channel)
        resnet152model = resnet152()
        self.layer3 = resnet152model.layer3
        self.layer4 = resnet152model.layer4
        self.avgpool = resnet152model.avgpool
        self.fc = resnet152model.fc

    def forward(self,x):
        decode_feature = self.bottleneck(x)
        output = self.layer3(decode_feature)
        output = self.layer4(output)
        output = self.avgpool(output)
        output = torch.squeeze(output)
        output = self.fc(output)
        return output

class CustomVGG(nn.Sequential):
    def __init__(self, bottleneck, short_module_names, org_vgg):
        module_dict = OrderedDict()
        module_dict['bottleneck'] = bottleneck
        short_module_set = set(short_module_names)
        for child_name, child_module in org_vgg.named_children():
            if child_name in short_module_set:
                if child_name == 'classifier':
                    module_dict['flatten'] = nn.Flatten(1)
                module_dict[child_name] = child_module

        super().__init__(module_dict)

class VGG_Layered(nn.Module):
    def __init__(
        self, vgg_orig: nn.Module, n_conv_per_block: list
    ) -> None:
        super().__init__()
        self.n_conv_per_block = n_conv_per_block
        for block_idx in range(len(self.n_conv_per_block)):
            exec(f'self.block{block_idx} = OrderedDict()')
        block_idx = 0
        conv_cnt = 0
        module_idx = 0
        for module in vgg_orig.features.children():
            if isinstance(module, torch.nn.Conv2d):
                conv_cnt += 1
                if conv_cnt > self.n_conv_per_block[block_idx]:
                    block_idx += 1
                    conv_cnt = 1
                    module_idx = 0                    
            exec(f"self.block{block_idx}['{module_idx}'] = module")
            module_idx += 1
        for block_idx in range(len(self.n_conv_per_block)):
            exec(f'self.block{block_idx} = torch.nn.Sequential(self.block{block_idx})')
        for child_name, child in vgg_orig.named_children():
            if child_name != 'features':
                exec(f'self.{child_name} = child')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # for block_idx in range(len(self.n_conv_per_block)):
        #     exec(f'x = self.block{block_idx}(x)')
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def custom_vgg16(bottleneck_channel=9, short_module_names=None, **kwargs):
    if short_module_names is None:
        short_module_names = ['block3', 'block4', 'avgpool', 'classifier']

    bottleneck = Bottleneck(bottleneck_channel, backbone='vgg')
    org_model = vgg16_bn(**kwargs)
    org_model = VGG_Layered(org_model, [2, 2, 3, 3, 3])
    return CustomVGG(bottleneck, short_module_names, org_model)

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
    
class SplitVggHead(nn.Module):
    def __init__(self,vgg,mean=(0.485,0.456,0.406),std=(0.229, 0.224, 0.225)) -> None:
        super().__init__()
        self.model = vgg.features[:8]
        self.mean = torch.tensor(mean).view(3,1,1)
        self.std = torch.tensor(std).view(3,1,1)
    def forward(self,x):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        x = (x - self.mean) / self.std
        return self.model(x)

class SplitVggTail(nn.Module):
    def __init__(self,vgg) -> None:
        super().__init__()
        self.features = vgg.features[8:]
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier
    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = nn.Flatten()(x)
        x = self.classifier(x)
        return x
    
class SplitResNetHead(nn.Module):
    def __init__(self,resnet,mean=(0.485,0.456,0.406),std=(0.229, 0.224, 0.225)) -> None:
        super().__init__()
        model = resnet
        sequential = OrderedDict()
        tail_dict = ['dqt','layer3','layer4','avgpool','flatten','fc']
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
    def __init__(self,resnet) -> None:
        super().__init__()
        model = resnet
        sequential = OrderedDict()
        tail_dict = ['dqt','layer3','layer4','avgpool','fc']
        for child_name, child_module in model.named_children():
            if child_name in tail_dict:
                if child_name == 'fc':
                    sequential['flatten'] = nn.Flatten(1)
                sequential[child_name] = child_module
        self.tail = nn.Sequential(sequential)
        
    def forward(self,x):
        return self.tail(x)

class Quantize(nn.Module):
    def __init__(self,qt=True) -> None:
        super().__init__()
        self.qt = qt
    def forward(self,x):
        return x.half() if self.qt else x.float()

def quantize_resnet(resnet):
    sequential = OrderedDict()
    for child_name, child_module in resnet.named_children():
        if child_name == 'layer3':
            sequential['qt'] = Quantize(qt=True)
            sequential['dqt'] = Quantize(qt=False)
        if child_name == 'fc':
            sequential['flatten'] = nn.Flatten(1)
        sequential[child_name] = child_module
    return nn.Sequential(sequential)
    
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
    
class SplitSC2Model(nn.Module):
    def __init__(self, pretrained_model, model_type, head=True, head_layers=None, tail_layers=None) -> None:
        super().__init__()
        if head_layers is None:
            head_layers=['bottleneck_layer.encoder']
        if tail_layers is None:
            tail_layers=['bottleneck_layer.decoder', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
        if head:
            self.mean = torch.tensor((0.485,0.456,0.406)).view(3, 1, 1)
            self.std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
            layers = head_layers
        else:
            self.mean = None
            self.std = None
            layers = tail_layers
        
        self.model = OrderedDict()
        for layer in layers:
            layer_name = layer if '.' not in layer else layer.split('.')[-1]
            if layer_name == 'fc':
                self.model['flatten'] = nn.Flatten(1)
            self.model[layer_name] = pretrained_model.get_submodule(layer)
        
        self.model = nn.Sequential(self.model)
        self.model_type = model_type
        if self.model_type == 'jpeg_feature':
            self.codec_encoder_decoder = pretrained_model.codec_encoder_decoder
        if self.model_type in ['cr-bq', 'end-to-end', 'entropic-student']:
            self.bottleneck_layer = pretrained_model.bottleneck_layer
        self.head = head

    def forward(self, x):
        device = x.device
        if self.mean is not None and self.std is not None:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
            x = (x - self.mean) / self.std
        if self.model_type == 'jpeg_feature' and not self.head:
            tmp_list = list()
            for sub_x in x:
                if self.codec_encoder_decoder is not None:
                    sub_x, file_size = self.codec_encoder_decoder(sub_x)
                tmp_list.append(sub_x)
            x = torch.stack(tmp_list,dim=0).to(device)
        if self.model_type == 'cr-bq' and not self.head:
            x = self.bottleneck_layer.decompressor(x)
        if self.model_type in ['end-to-end', 'entropic-student'] and not self.head:
            x = self.bottleneck_layer.entropy_bottleneck.decompress(self.bottleneck_layer.entropy_bottleneck.compress(x), x.size()[-2:])
        x = self.model(x)
        if self.model_type == 'cr-bq' and self.head:
            x = self.bottleneck_layer.compressor(x)
        return x
    