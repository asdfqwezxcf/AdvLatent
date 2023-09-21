import torch
from torchvision.models import resnet50, resnet152,vgg16_bn
from torchvision import datasets, transforms
from src.GradientAttack import FGSM,BIM,MI_FGSM,PGD
from src.models import CustomResNet152,SplitHead,SplitTail,AdvModel,SplitResNetHead,SplitResNetTail, SplitSC2Model,SplitVggHead,SplitVggTail
from src.models import custom_vgg16,quantize_resnet
from src.DecisionAttack import Evolutionary,HSJA,SignOPT,TriAttack
from src.ScoreAttack import Nattack, NES, SquareAttack
import logging
import numpy as np
import argparse
import random
from importlib import reload
from multiprocessing import cpu_count
from src.backbone import get_backbone, check_if_updatable
from src.wrapper import get_wrapped_classification_model

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

def arguments_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-md', '--model', type=str, metavar='', default='resnet50',
                        help = 'the model for experiments (vgg16, resnet50, resnet152, supervised, distill, bottlefit, vgg16_bottlefit, jpeg_feature, cr-bq, end-to-end, entropic-student)')
    parser.add_argument('-ch', '--channel', type=int, metavar='', default=3,
                        help = 'the channels of the head output which representing different compression ratio')
    parser.add_argument('-s', '--split', action='store_true',
                        help='attack the splited model (tail only) or the whole model (head + tail)')
    parser.add_argument('-atk', '--attack', type=str, metavar='',default='FGSM',
                        help='specify the attack algorithm (FGSM, BIM, MIM, PGD, PGD_2, Natk, NES, Evo, Sopt, HSJA, Satk, Tatk)')
    parser.add_argument('-eps', '--epsilon', type=float, metavar='',default=0.1,
                        help='the maximum allowed perturbation')
    parser.add_argument('-t', '--target', type=int, metavar='', default=-1,
                        help='specify target: -1 means non-targeted attack (others are not implemented yet)')
    parser.add_argument('-log', '--logging',type=str,metavar='',default='default.log',
                        help='specify name of the log file')
    parser.add_argument('-d', '--device', type=int, metavar='', default=0,
                        help='specify the gpu device, -1 means cpu')
    parser.add_argument('-b', '--batchsize', type=int, metavar='', default=64,
                        help='specify the batchsize')
    parser.add_argument('-ns', '--nsamples', type=int, metavar='', default=1000,
                        help='specify the number of samples in experiments.')
    
    return parser.parse_args()

def setup_log(log):
    for logname in logging.Logger.manager.loggerDict.keys():
        logger = logging.getLogger(name=logname)
        logger.disabled = True # clear uneccessary log in torchdistill and sc2bench package 
    reload(logging)

    explog = logging.getLogger(name='exp_record')
    explog.setLevel(logging.INFO)
    handler = logging.FileHandler('log/'+log)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    explog.addHandler(handler)

    return explog

def benchmarking():
    args = arguments_parser()
    assert args.model in ['vgg16','resnet50','resnet152', 'supervised', 'distill', 'bottlefit', 'vgg16_bottlefit', 'jpeg_feature', 'cr-bq', 'end-to-end', 'entropic-student', 'resnet-qt'], "the model is not defined"
    assert args.attack in ['FGSM', 'BIM', 'MIM', 'PGD', 'PGD_2', 'Natk', 'NES', 'Evo', 'Sopt', 'HSJA', 'Satk','Tatk'], "the attack is not defined"
    explog = setup_log(args.logging)
    # load model
    clip_min = 0.
    clip_max = 1.
    headmodel = None
    ckpt = None
    if args.model == 'vgg16':
        try:
            model = vgg16_bn(weights="IMAGENET1K_V1")
        except:
            model = vgg16_bn(pretrained=True) # backward compatible to older torch version
    elif args.model == 'resnet152':
        try:
            model = resnet152(weights="IMAGENET1K_V1")
        except:
            model = resnet152(pretrained=True)
    elif args.model in ['resnet50', 'resnet-qt']:
        try:
            model = resnet50(weights="IMAGENET1K_V1")
        except:
            model = resnet50(pretrained=True)
        if args.model == 'resnet-qt':
            model = quantize_resnet(model)
    elif args.model == 'vgg16_bottlefit':
        try:
            model = custom_vgg16(weights='VGG16_BN_Weights.IMAGENET1K_V1')
        except:
            model = custom_vgg16(pretrained=True)
        ckpt = torch.load(f'resource/bottlefit/vgg16_bottlefit_9ch.pt')['model']
    elif args.model == 'jpeg_feature':
        config = {'name': 'CodecFeatureCompressionClassifier', 'params': {'codec_params': [{'type': 'PILTensorModule', 'params': {'format': 'JPEG', 'quality': 95, 'returns_file_size': True}}], 'encoder_config': {'sequential': ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2']}, 'decoder_config': {'sequential': ['layer3', 'layer4', 'avgpool']}, 'classifier_config': {'sequential': ['fc']}, 'post_transform_params': None, 'analysis_config': {'analyzer_configs': [{'type': 'FileSizeAccumulator', 'params': {'unit': 'KB'}}]}}, 'classification_model': {'name': 'resnet50', 'params': {'num_classes': 1000, 'weights': 'ResNet50_Weights.IMAGENET1K_V1'}}}
        try:
            model = get_wrapped_classification_model(config)
        except:
            config['classification_model'] = {'name': 'resnet50', 'params': {'num_classes': 1000, 'pretrained': True}}
            model = get_wrapped_classification_model(config)
    elif args.model in ['cr-bq', 'end-to-end', 'entropic-student']:
        config = None
        if args.model == 'cr-bq':
            config = {'num_classes': 1000, 'weights': 'ResNet50_Weights.IMAGENET1K_V1', 'bottleneck_config': {'name': 'larger_resnet_bottleneck', 'params': {'bottleneck_channel': 12, 'bottleneck_idx': 12, 'output_channel': 256, 'compressor_transform_params': [{'type': 'SimpleQuantizer', 'params': {'num_bits': 16}}], 'decompressor_transform_params': [{'type': 'SimpleDequantizer', 'params': {'num_bits': 8}}]}}, 'resnet_name': 'resnet50', 'pre_transform_params': None, 'skips_avgpool': False, 'skips_fc': False, 'analysis_config': {'analyzes_after_compress': True, 'analyzer_configs': [{'type': 'FileSizeAnalyzer', 'params': {'unit': 'KB'}}]}}
            ckpt = torch.load(f"resource/ghnd-bq/ilsvrc2012-resnet50-bq12ch_from_resnet50.pt", map_location='cpu')['model']
        elif args.model == 'end-to-end':
            config = {'num_classes': 1000, 'weights': 'ResNet50_Weights.IMAGENET1K_V1', 'bottleneck_config': {'name': 'FPBasedResNetBottleneck', 'params': {'num_bottleneck_channels': 24, 'num_target_channels': 256}}, 'resnet_name': 'resnet50', 'pre_transform_params': None, 'skips_avgpool': False, 'skips_fc': False, 'analysis_config': {'analyzes_after_compress': True, 'analyzer_configs': [{'type': 'FileSizeAnalyzer', 'params': {'unit': 'KB'}}]}}
            ckpt = torch.load(f"resource/end-to-end/ilsvrc2012-splittable_resnet50-fp-beta1.024e-7.pt", map_location='cpu')['model']
        elif args.model == 'entropic-student':
            config = {'num_classes': 1000, 'weights': 'ResNet50_Weights.IMAGENET1K_V1', 'bottleneck_config': {'name': 'FPBasedResNetBottleneck', 'params': {'num_bottleneck_channels': 24, 'num_target_channels': 256}}, 'resnet_name': 'resnet50', 'pre_transform_params': None, 'skips_avgpool': False, 'skips_fc': False, 'analysis_config': {'analyzes_after_compress': True, 'analyzer_configs': [{'type': 'FileSizeAnalyzer', 'params': {'unit': 'KB'}}]}}
            ckpt = torch.load(f"resource/entropic_student/ilsvrc2012-splittable_resnet50-fp-beta0.08_from_resnet50.pt", map_location='cpu')['model']
        if config is not None:
            try:
                model = get_backbone('splittable_resnet', **config)
            except:
                config['pretrained'] = True
                config.pop('weights') # backward compatible to older torch version
                model = get_backbone('splittable_resnet', **config)
    else:
        model = CustomResNet152(args.channel)
        if args.model == 'supervised':
            model = torch.load('resource/cross_entropy/naive_%dch.pth'%args.channel,map_location='cpu')
        if args.model == 'distill':
            model = torch.load('resource/knowledge_distillation/distill_%dch.pth'%args.channel,map_location='cpu')   
        else:
            model = torch.load('resource/bottlefit/bottlefit_%dch.pth'%args.channel,map_location='cpu')

    if ckpt is not None:
        model.load_state_dict(ckpt, strict=True)
    
    if check_if_updatable(model):
        model.update()

    tailmodel = AdvModel(model)

    # split model
    if args.split:
        clip_min = None # for latent representation, there is no explicit lower bound and upper bound
        clip_max = None
        if args.model == 'vgg16':
            headmodel = SplitVggHead(model)
            tailmodel = SplitVggTail(model)
        elif args.model in ['resnet50', 'resnet152', 'resnet-qt']:
            headmodel = SplitResNetHead(model)
            tailmodel = SplitResNetTail(model)
        elif args.model in ['vgg16_bottlefit', 'jpeg_feature', 'cr-bq', 'end-to-end', 'entropic-student']:
            head_layers = None
            tail_layers = None
            if args.model == 'jpeg_feature':
                head_layers = ['encoder']
                tail_layers = ['decoder', 'fc']
            if args.model == 'vgg16_bottlefit':
                head_layers=['bottleneck.encoder']
                tail_layers=['bottleneck.decoder', 'block3', 'block4', 'avgpool', 'flatten', 'classifier']
            headmodel = SplitSC2Model(model, args.model, head=True, head_layers=head_layers, tail_layers=tail_layers)
            tailmodel = SplitSC2Model(model, args.model, head=False, head_layers=head_layers, tail_layers=tail_layers)
        else:
            headmodel = SplitHead(model)
            tailmodel = SplitTail(model)
    
    # set up device
    device = "cuda:%d"%args.device if args.device>=0 else "cpu"

    tailmodel = tailmodel.to(device)
    tailmodel.eval()
    if headmodel is not None:
        headmodel = headmodel.to(device)
        headmodel.eval()

    # set up attacks
    batchsize = args.batchsize # some attacks use broadcasting and requires smaller batchsize due to the memory constraint
    target = True
    if args.target < 0:
        target = False
    # gradient based
    if args.attack == 'FGSM':
        atk = FGSM(epsilon=args.epsilon,clip_min=clip_min,clip_max=clip_max,target=target)
    elif args.attack == 'BIM':
        atk = BIM(epsilon=args.epsilon,clip_min=clip_min,clip_max=clip_max,target=target)
    elif args.attack == 'MIM':
        atk = MI_FGSM(epsilon=args.epsilon,clip_min=clip_min,clip_max=clip_max,target=target)
    elif args.attack == 'PGD':
        atk = PGD(epsilon=args.epsilon,clip_min=clip_min,clip_max=clip_max,target=target)
    elif args.attack == 'PGD_2':
        atk = PGD(epsilon=args.epsilon,clip_min=clip_min,clip_max=clip_max,target=target,ord=2)
    # score based
    elif args.attack == 'Natk':
        atk = Nattack(epsilon=args.epsilon,clip_min=clip_min,clip_max=clip_max,target=target)
    elif args.attack == 'NES':
        atk = NES(epsilon=args.epsilon,clip_min=clip_min,clip_max=clip_max,target=target)
    elif args.attack == 'Satk':
        atk = SquareAttack(epsilon=args.epsilon,clip_min=clip_min,clip_max=clip_max,target=target)
    # decision based
    elif args.attack == 'Evo':
        atk = Evolutionary(epsilon=args.epsilon,clip_min=clip_min,clip_max=clip_max,target=target)
    elif args.attack == 'Sopt':
        atk = SignOPT(epsilon=args.epsilon,clip_min=clip_min,clip_max=clip_max,target=target)
    elif args.attack == 'HSJA':
        atk = HSJA(epsilon=args.epsilon,clip_min=clip_min,clip_max=clip_max,target=target)
    elif args.attack == 'Tatk':
        atk = TriAttack(epsilon=args.epsilon,clip_min=clip_min,clip_max=clip_max,target=target)
    
    # set up dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageNet('dataset/imagenet', split='val', transform=transforms.Compose([
                transforms.Resize((256,256)),
		        transforms.CenterCrop((224,224)),
		        transforms.ToTensor(),
                ])),
        batch_size=batchsize, shuffle=True, num_workers=cpu_count())

    # start recording and run experiment
    explog.info("Experiments start! - Model: %s_%dch, Split: %s, Target: %d, Attack: %s, epsilon: %.3f"%(args.model,args.channel,args.split,args.target,args.attack,args.epsilon))
    
    success = 0
    nsamples = 0
    max_samples = args.nsamples
    for i, samples in enumerate(test_loader):
        nsample_total = (i+1)*batchsize
        if nsamples >= max_samples:
            break
        x,y = samples
        # set up x
        x = x.to(device)
        y = y.to(device)
        if headmodel is not None:
            with torch.no_grad():
                x = headmodel(x).detach()
        with torch.no_grad():
            y_pred = tailmodel(x)
        # report baseline acc if epsilon is 0
        if args.epsilon == 0.:
            success += (y_pred.argmax(1) == y).type(torch.float).sum().item()
            print("Accuracy without attack: %.6f"%(success/nsample_total),end='\r')
        else:
            if len(y_pred.shape) == 1: # expand dim if batchsize is 1
                y_pred = y_pred[None,:]
            mask = torch.argmax(y_pred,dim=1) == y # only attack the correct samples
            x = x[mask]
            y = y[mask]
            if y.size(dim=0) == 0:
                continue
            if max_samples-nsamples < y.size(dim=0): # make sure the test samplesize == maximum sample size
                x = x[:max_samples-nsamples]
                y = y[:max_samples-nsamples]
                nsamples = max_samples
            else:
                nsamples += y.size(dim=0)
            # set up y
            if target:
                if args.target != 1000:
                    y = y.new_full(y.shape,args.target).to(device) # target attack
                else:
                    y = y_pred[mask]
                    y = torch.argmin(y,dim=1).to(device) # least likely attack
            
            # run adv algos
            x_adv = atk(tailmodel,x.float(),y)
            # collect metrics
            with torch.no_grad():
                y_pred = tailmodel(x_adv)
            if len(y_pred.shape) == 1:
                y_pred = y_pred[None,:]
            if target:
                success += (y_pred.argmax(1) == y).type(torch.float).sum().item()
            else:
                success += (y_pred.argmax(1) != y).type(torch.float).sum().item()
            print("Attack success rate: %.6f"%(success/nsamples),end='\r')
    if args.epsilon == 0.:
        explog.info("Accuracy without attack: %.6f"%(success/nsample_total))
        print("Accuracy without attack: %.6f"%(success/nsample_total))
    else:    
        explog.info("Attack success rate: %.6f"%(success/nsamples))
        print("Attack success rate: %.6f"%(success/nsamples))


if __name__ == '__main__':
    benchmarking()    
