# Benchmarking the Adversaral Robustness in Latent Representations

## Environment
```bash
conda env create -n AdvSC --file requirements.yml
conda activate AdvSC
```

## Dataset
Experiments are based on ImageNet2012 dataset. To preproduce the results, you should download the [IMAGENET](https://image-net.org/download) first. And put it in the right directory.
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
  - ResNet152-Vanilla, ResNet152-Bottleneck
  - ResNet50-vanilla, ResNet50-Bottleneck
  - VGG16-Vanilla, VGG16-Bottleneck
### 2. Attacks on different feature compression approach
  - One-stage Training
    - supervised compression (SC)
    - knowledge distillation (KD)
  - Multi-stage Training
    - BottleFit (BF)
    - Entropic Student (ES)
  - Other Feature Compression
    - Bit Quantization (QT)
    - JEPG-based Codec (JC)
### 3. Attacks on different compression ratio
  - Resnet152 -org, -ch3, -ch12

To run experiments, you should first download our pretrained model from [Here](https://drive.google.com/file/d/1t_BJih8nyuRhxUkqHxrYwBYamM4HgphP/view?usp=drive_link) and make sure they are in the right directory.
```bash
cd ~/SCAR-Benchmarking-the-Split-Computing-Adversarial-Robustness
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
