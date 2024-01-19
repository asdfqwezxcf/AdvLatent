import torch
from torchvision.models import resnet50
from torchvision import datasets, transforms
from adaptive.adapt import Adpt
from adaptive.model_wrapper import *
import logging
import numpy as np
import argparse
import random
from importlib import reload
from multiprocessing import cpu_count

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

def arguments_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-md', '--model', type=str, metavar='', default='resnet50',
                        help = 'the model for experiments (resnet50, dat,fastat)')
    parser.add_argument('-s', '--split', action='store_true',
                        help='attack the splited model (tail only) or the whole model (head + tail)')
    parser.add_argument('-sc', '--scale', type=float, metavar='',default=0.3,
                        help='the scale of distance regularization')
    parser.add_argument('-eps', '--epsilon', type=float, metavar='',default=0.1,
                        help='the maximum allowed perturbation')
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
    explog = setup_log(args.logging)

    clip_min = 0.
    clip_max = 1.
    headmodel = None

    try:
        model = resnet50(weights="IMAGENET1K_V1")
    except:
        model = resnet50(pretrained=True)
    if args.model == 'dat':
        model.load_state_dict(torch.load('resource/dat/datpgd_imagenet_epoch30.checkpoint')['state_dict'])
    if args.model == 'fastat':
        checkpoint = torch.load("resource/fat/imagenet_model_weights_4px.pth.tar")['state_dict']
        sd = {k[len('module.'):]:v for k,v in checkpoint.items()}
        model.load_state_dict(sd)

    # split model
    if args.split:
        clip_min = None # for latent representation, there is no explicit lower bound and upper bound
        clip_max = None
        headmodel = SplitResNetHead(model)
        if args.model == 'dat':
            headmodel = AdvWrapper(headmodel,mean=(0,0,0),std=(1,1,1))
        else:
            headmodel = AdvWrapper(headmodel)
        tailmodel = SplitResNetTail(model)
        tailmodel = LatentWrapper(tailmodel)
    else:
        tailmodel = LatentWrapper(model)
        if args.model == 'dat':
            tailmodel = AdvWrapper(tailmodel,(0,0,0),(1,1,1))
        else:
            tailmodel = AdvWrapper(tailmodel)

    # set up device
    device = "cuda:%d"%args.device if args.device>=0 else "cpu"

    tailmodel = tailmodel.to(device)
    tailmodel.eval()
    if headmodel is not None:
        headmodel = headmodel.to(device)
        headmodel.eval()

    # set up attacks
    batchsize = args.batchsize # some attacks use broadcasting and requires smaller batchsize due to the memory constraint
    target = False
    atk = Adpt(epsilon=args.epsilon,clip_min=clip_min,clip_max=clip_max,target=target,scale=args.scale,ord=2)

    # set up dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageNet('dataset/imagenet', split='val', transform=transforms.Compose([
                transforms.Resize((256,256)),
		        transforms.CenterCrop((224,224)),
		        transforms.ToTensor(),
                ])),
        batch_size=batchsize, shuffle=True, num_workers=cpu_count())
    if args.model == 'fastat':
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageNet('dataset/imagenet', split='val', transform=transforms.Compose([
                    transforms.RandomResizedCrop(288),
                    transforms.ToTensor(),
                    ])),
            batch_size=batchsize, shuffle=True, num_workers=cpu_count())
        
    # start recording and run experiment
    explog.info("Experiments start! - Model: %s, Split: %s, Adaptive Attack scale: %.3f, epsilon: %.3f"%(args.model,args.split,args.scale,args.epsilon))
    
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
            _, y_pred = tailmodel(x)

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
            
            # run adv algos
            x_adv = atk(tailmodel,x,y)

            # collect metrics
            with torch.no_grad():
                _, y_pred = tailmodel(x_adv)
            if len(y_pred.shape) == 1:
                y_pred = y_pred[None,:]
            success += (y_pred.argmax(1) != y).type(torch.float).sum().item()
            print("Attack success rate: %.6f, number of samples: %d"%(success/nsamples,nsamples),end='\r')
    if args.epsilon == 0.:
        explog.info("Accuracy without attack: %.6f"%(success/nsample_total))
        print("Accuracy without attack: %.6f"%(success/nsample_total))
    else:  
        explog.info("Attack success rate: %.6f"%(success/nsamples))
        print("Attack success rate: %.6f"%(success/nsamples))

if __name__ == '__main__':
    benchmarking()    
    