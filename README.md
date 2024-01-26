# Benchmarking the Adversaral Robustness in Latent Representations

## Environment
```bash
conda env create -n AdvLatent --file requirements.yml
conda activate AdvLatent
```

## Dataset
Experiments are based on ImageNet2012 dataset. To reproduce the results, you should download the [IMAGENET](https://image-net.org/download) first. And put it in the right directory.
```bash
cd ~/AdvLatent
mkdir dataset/imagenet
cd dataset/imagenet
wget ${Imagenet_2012_url}
```

## Attack Pool
- **Gradient-based Attacks**: A series of white-box attacks, now support for Linf distance
  - Fast Gradient Sign Method (FGSM): [Paper](https://arxiv.org/abs/1412.6572)
  - Basic Iterative Method (BIM): [Paper](https://arxiv.org/abs/1607.02533)
  - Momentum Iterative Method (MIM): [Paper](https://arxiv.org/abs/1710.06081)
  - Project Gradient Descent (PGD): [Paper](https://arxiv.org/abs/1706.06083)
- **Score-based Attacks**: A series of black-box attacks only use soft labels (logits), now support for Linf distance
  - Natural Evolutionary Strategies (NES): [Paper](https://arxiv.org/abs/1804.08598)
  - N-Attack (Natk): [Paper](https://arxiv.org/abs/1905.00441)
  - Square Attack (Satk): [Paper](https://arxiv.org/abs/1912.00049)
- **Decision-based Attacks**: A series of black-box attacks only use the hard label (the largest logit), now support for L2 distance
  - Evolutionary Attack (Evo): [Paper](https://arxiv.org/abs/1904.04433)
  - Hope-Skip-Jump Attack (HSJA): [Paper](https://arxiv.org/abs/1904.02144)
  - Sign-OPT Attack (Sopt): [Paper](https://arxiv.org/abs/1909.10773)
  - Triangle Attack (Tatk): [Paper](https://arxiv.org/abs/2112.06569)

## Model Pool
### 1. Attacks on different model architecture
  - ResNet152, ResNet152-fc
  - ResNet50, ResNet50-fc
  - VGG16, VGG16-fc
### 2. Attacks on different feature compression approach
  - Dimension Compression
    - supervised compression (SC)
    - knowledge distillation (KD)
    - BottleFit (BF)
  - Data Compression
    - Bit Quantization (QT)
    - JEPG-based Codec (JC)
  - Dimension Compression + Data Compression
    - Entropic Student (ES)
### 3. Attacks on different compression ratio
  - Resnet152 -org, -ch3, -ch12
### 4. Attacks on adversarial training
  - FastAT: [Paper](https://arxiv.org/abs/2001.03994)
  - DAT: [Paper](https://proceedings.mlr.press/v180/zhang22a/zhang22a.pdf)
  
To run experiments, you should first download pretrained models from [Here](https://drive.google.com/file/d/11r8cslaRaBrZL2klQCmp4epRlloMnz-n/view?usp=sharing) and make sure they are in the right directory.
```bash
cd ~/AdvLatent
wget ${resource.tar}
tar -xvf resource.tar
```

## Code Usage

```bash
python run_benchmark.py -h
usage: run_benchmark.py [-h] [-md] [-ch] [-s] [-atk] [-eps] [-t] [-log] [-d]
                        [-b] [-ns]

optional arguments:
  -h, --help         show this help message and exit
  -md , --model      the model for experiments (vgg16, resnet50, resnet152,
                     supervised, distill, bottlefit, vgg16_bottlefit,
                     jpeg_feature, cr-bq, end-to-end, entropic-student)
                     (default: resnet50)
  -ch , --channel    the channels of the head output which representing
                     different compression ratio (default: 3)
  -s, --split        attack the splited model (tail only) or the whole model
                     (head + tail) (default: False)
  -atk , --attack    specify the attack algorithm (FGSM, BIM, MIM, PGD, PGD_2,
                     Natk, NES, Evo, Sopt, HSJA, Satk, Tatk) (default: FGSM)
  -eps , --epsilon   the maximum allowed perturbation (default: 0.1)
  -t , --target      specify target: -1 means non-targeted attack (others are
                     not implemented yet) (default: -1)
  -log , --logging   specify name of the log file (default: default.log)
  -d , --device      specify the gpu device, -1 means cpu (default: 0)
  -b , --batchsize   specify the batchsize (default: 256)
  -ns , --nsamples   specify the number of samples in experiments. (default:
                     1000)
```

