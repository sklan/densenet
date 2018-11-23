# Densely Connected Convolutional Networks

This repository contains an implementation of DenseNet in Keras. 
DenseNet is a neural network architecture for object recognition.
DenseNet were introduced in the paper [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf) which won the CVPR 2017 best paper award.

Recent advances in computer hardware has allowed researchers to train very deep neural network models. 
However, as information passes through many layers, it can vanish. 
To address this problem of vanishing gradient the key idea is to connect early layers to the later layers. 
DenseNet takes this simple idea further and connects all layers directly with each other. 
This enables maximum information between the layers.

Pretrained Models:


| Model         |  L |  K | C10  | C10+ |
|---------------|:--:|:--:|------|------|
| [DenseNet]()  | 40 | 12 | 90.5 | 92.0 |
| [DenseNetC]() | 40 | 12 | 91.7 | 92.0 |
