# Densely Connected Convolutional Networks

This repository contains an implementation of DenseNet in Keras. 
DenseNet is a neural network architecture for object recognition.
DenseNet was introduced in the paper [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf) which won the CVPR 2017 best paper award.

Recent advances in computer hardware has allowed researchers to train very deep neural network models. 
However, as information passes through many layers, it can vanish. 
To address this problem of vanishing gradient the key idea is to connect early layers to the later layers. 
DenseNet takes this simple idea further and connects all layers directly with each other. 
This enables maximum information flow between the layers.

## Pretrained Models:


| Model         |  L |  K | C10  | C10+ |
|---------------|:--:|:--:|------|------|
| DenseNet  | 40 | 12 | [92.90](https://github.com/Sklan/densenet/blob/master/CIFAR10/DenseNet4012_CIFAR10.h5) | [93.03](https://github.com/Sklan/densenet/blob/master/CIFAR10/DenseNet4012_CIFAR10+.h5)|
| DenseNetC | 40 | 12 | [91.70](https://github.com/Sklan/densenet/blob/master/CIFAR10/DenseNetC4012_CIFAR10.h5) | ---- |


## Arguments
--augment type=bool, default=False, Augment data or not.

--batch_size type=int, default=64, Sets the batch size for training.

--bottleneck type=bool, default=False, Add a bottleneck layer.

--depth type=int, default=40, Sets the depth of the model.

--dropout_rate type=float, default=0.2, Sets the dropout to be applied.

--epochs type=int, default=300, Sets the batch size for training.

--growth_rate type=int, default=12, Sets the growth rate.

--learning_rate type=float, default=0.1, Sets the learning rate.

--num_classes type=int, default=10, Set the number of classes.

--reduction type=float, default=1.0, Sets the reduction to be applied.

--weight_decay type=float, default=1e-4, Sets the weight decay to be applied.

