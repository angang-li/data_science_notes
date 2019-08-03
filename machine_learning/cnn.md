# Convolutional Neural Networks

- [Convolutional Neural Networks](#convolutional-neural-networks)
  - [1. Foundations of convolutional neural networks (CNN)](#1-foundations-of-convolutional-neural-networks-cnn)
    - [1.1. Computer vision](#11-computer-vision)
    - [1.2. Convolution operation](#12-convolution-operation)
    - [1.3. Padding](#13-padding)
    - [1.4. Strided convolution](#14-strided-convolution)
    - [1.5. Convolution layer](#15-convolution-layer)
    - [1.6. Pooling layer](#16-pooling-layer)
    - [1.7. CNN schematics](#17-cnn-schematics)
    - [1.8. Why convolutions](#18-why-convolutions)
  - [2. CNN case studies](#2-cnn-case-studies)
    - [2.1. Classic networks](#21-classic-networks)
    - [2.2. ResNet](#22-resnet)
    - [2.3. Inception neural networks](#23-inception-neural-networks)
  - [3. Practical advices for using CNN](#3-practical-advices-for-using-cnn)
    - [3.1. General tips](#31-general-tips)
    - [3.2. Data vs. hand-engineering](#32-data-vs-hand-engineering)
    - [3.3. Tips for doing well on benchmarks/winning competitions](#33-tips-for-doing-well-on-benchmarkswinning-competitions)

## 1. Foundations of convolutional neural networks (CNN)

### 1.1. Computer vision

- Why computer vision

  - Rapid advances in computer vision are enabling brand new applications
  - Cross-fertilization into other areas, because the computer vision research community has been so creative and so inventive in coming up with new neural network architectures and algorithms

- Computer vision applications

  - Image classification
  - Object detection
  - Neural style transfer

- One challenge of computer vision: the inputs can get really big

  - Easily overfitting
  - Infeasible computational and memory requirements

### 1.2. Convolution operation

- How convolution operation works

  <img src="Resources/deep_learning/cnn/convolution_operation.png" width=600> <br>
  Thechnically, this operation is called cross correlation by mathematicians

- Code

  - Python: `conv_forward`
  - TensorFlow: `tf.nn.conv2d`
  - Keras: `Conv2D`

- Edge detection, an example application of convolution

  - Edge example

    <img src="Resources/deep_learning/cnn/edge_example.png" width=400>

  - Vertical edge detection

    <img src="Resources/deep_learning/cnn/vertical_edge_detection.png" width=400>

  - Hand-coded edge detection filters

    <img src="Resources/deep_learning/cnn/edge_detection_filters.png" width=400>

    Dimension: <a href="https://www.codecogs.com/eqnedit.php?latex=f&space;\times&space;f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f&space;\times&space;f" title="f \times f" /></a>, where f is usually odd

  - Learn the filter parameters from neural network

    <img src="Resources/deep_learning/cnn/edge_detection_parameterize.png" width=150>

### 1.3. Padding

- Downside of convolution operation

  - (-) Shrinking output
  - (-) throwing away information from edges

- Padding on the input matrix

  Common to zero-pad the border. In the example below, the padding p = 1.

  <img src="Resources/deep_learning/cnn/keras_conv2d_padding.gif" width=450>

- Valid and same convolutions

  - **Valid convolution:** no padding

    <a href="https://www.codecogs.com/eqnedit.php?latex=n\times&space;n&space;\&space;image&space;\&space;\&space;\ast&space;\&space;\&space;f&space;\times&space;f&space;\&space;filter&space;\&space;\&space;\rightarrow&space;\&space;\&space;(n-f&plus;1)&space;\times&space;(n-f&plus;1)&space;\&space;output" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n\times&space;n&space;\&space;image&space;\&space;\&space;\ast&space;\&space;\&space;f&space;\times&space;f&space;\&space;filter&space;\&space;\&space;\rightarrow&space;\&space;\&space;(n-f&plus;1)&space;\times&space;(n-f&plus;1)&space;\&space;output" title="n\times n \ image \ \ \ast \ \ f \times f \ filter \ \ \rightarrow \ \ (n-f+1) \times (n-f+1) \ output" /></a>

  - **Same convolution:** pad so that output size is the same as the input sizes

    <a href="https://www.codecogs.com/eqnedit.php?latex=p=\frac{f-1}{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p=\frac{f-1}{2}" title="p=\frac{f-1}{2}" /></a>

- Advantages of padding

  - (+) Allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. This is important for building deeper networks, since otherwise the height/width would shrink as you go to deeper layers.
  - (+) Helps us keep more of the information at the border of an image. Without padding, very few values at the next layer would be affected by pixels as the edges of an image.

### 1.4. Strided convolution

- Stride example

  In the example below, stride s = 2.

  <img src="Resources/deep_learning/cnn/stride.png" width=450>

  Output dimension: <a href="https://www.codecogs.com/eqnedit.php?latex=\left&space;\lfloor&space;\frac{n&plus;2p-f}{s}&plus;1&space;\right&space;\rfloor&space;\&space;\&space;\times&space;\&space;\&space;\left&space;\lfloor&space;\frac{n&plus;2p-f}{s}&plus;1&space;\right&space;\rfloor" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left&space;\lfloor&space;\frac{n&plus;2p-f}{s}&plus;1&space;\right&space;\rfloor&space;\&space;\&space;\times&space;\&space;\&space;\left&space;\lfloor&space;\frac{n&plus;2p-f}{s}&plus;1&space;\right&space;\rfloor" title="\left \lfloor \frac{n+2p-f}{s}+1 \right \rfloor \ \ \times \ \ \left \lfloor \frac{n+2p-f}{s}+1 \right \rfloor" /></a> <br>
  By convention, the filter must be fully contained in the input image to do convolution.

### 1.5. Convolution layer

- Convolutions over volume

  <img src="Resources/deep_learning/cnn/convolution_over_volume.png" width=500>

  <a href="https://www.codecogs.com/eqnedit.php?latex=n\times&space;n&space;\times&space;n_c&space;\&space;\&space;\ast&space;\&space;\&space;f&space;\times&space;f&space;\times&space;n_c&space;\&space;\&space;\rightarrow&space;\&space;\&space;(n-f&plus;1)&space;\times&space;(n-f&plus;1)&space;\times&space;n_c'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n\times&space;n&space;\times&space;n_c&space;\&space;\&space;\ast&space;\&space;\&space;f&space;\times&space;f&space;\times&space;n_c&space;\&space;\&space;\rightarrow&space;\&space;\&space;(n-f&plus;1)&space;\times&space;(n-f&plus;1)&space;\times&space;n_c'" title="n\times n \times n_c \ \ \ast \ \ f \times f \times n_c \ \ \rightarrow \ \ (n-f+1) \times (n-f+1) \times n_c'" /></a>

- Notation

  - <a href="https://www.codecogs.com/eqnedit.php?latex=f^{(l)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f^{(l)}" title="f^{(l)}" /></a>: filter size
  - <a href="https://www.codecogs.com/eqnedit.php?latex=p^{(l)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p^{(l)}" title="p^{(l)}" /></a>: padding size
  - <a href="https://www.codecogs.com/eqnedit.php?latex=s^{(l)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s^{(l)}" title="s^{(l)}" /></a>: stride size

- Dimensions

  - Input: <a href="https://www.codecogs.com/eqnedit.php?latex=n_H^{(l-1)}&space;\times&space;n_W^{(l-1)}&space;\times&space;n_C^{(l-1)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_H^{(l-1)}&space;\times&space;n_W^{(l-1)}&space;\times&space;n_C^{(l-1)}" title="n_H^{(l-1)} \times n_W^{(l-1)} \times n_C^{(l-1)}" /></a>

  - Output: <a href="https://www.codecogs.com/eqnedit.php?latex=n_H^{(l)}&space;\times&space;n_W^{(l)}&space;\times&space;n_C^{(l)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_H^{(l)}&space;\times&space;n_W^{(l)}&space;\times&space;n_C^{(l)}" title="n_H^{(l)} \times n_W^{(l)} \times n_C^{(l)}" /></a>, where

      <a href="https://www.codecogs.com/eqnedit.php?latex=n_H^{(l)}&space;=&space;\left&space;\lfloor&space;\frac{n_H^{(l-1)}&space;&plus;&space;2p^{(l)}&space;-&space;f^{(l)}}{s^{(l)}}&plus;1&space;\right&space;\rfloor" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_H^{(l)}&space;=&space;\left&space;\lfloor&space;\frac{n_H^{(l-1)}&space;&plus;&space;2p^{(l)}&space;-&space;f^{(l)}}{s^{(l)}}&plus;1&space;\right&space;\rfloor" title="n_H^{(l)} = \left \lfloor \frac{n_H^{(l-1)} + 2p^{(l)} - f^{(l)}}{s^{(l)}}+1 \right \rfloor" /></a> <br>
      <a href="https://www.codecogs.com/eqnedit.php?latex=n_W^{(l)}&space;=&space;\left&space;\lfloor&space;\frac{n_W^{(l-1)}&space;&plus;&space;2p^{(l)}&space;-&space;f^{(l)}}{s^{(l)}}&plus;1&space;\right&space;\rfloor" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_W^{(l)}&space;=&space;\left&space;\lfloor&space;\frac{n_W^{(l-1)}&space;&plus;&space;2p^{(l)}&space;-&space;f^{(l)}}{s^{(l)}}&plus;1&space;\right&space;\rfloor" title="n_W^{(l)} = \left \lfloor \frac{n_W^{(l-1)} + 2p^{(l)} - f^{(l)}}{s^{(l)}}+1 \right \rfloor" /></a>

  - Each filter: <a href="https://www.codecogs.com/eqnedit.php?latex=f^{(l)}&space;\times&space;f^{(l)}&space;\times&space;n_C^{(l-1)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f^{(l)}&space;\times&space;f^{(l)}&space;\times&space;n_C^{(l-1)}" title="f^{(l)} \times f^{(l)} \times n_C^{(l-1)}" /></a>

      Weights: <a href="https://www.codecogs.com/eqnedit.php?latex=f^{(l)}&space;\times&space;f^{(l)}&space;\times&space;n_C^{(l-1)}&space;\times&space;n_C^{(l)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f^{(l)}&space;\times&space;f^{(l)}&space;\times&space;n_C^{(l-1)}&space;\times&space;n_C^{(l)}" title="f^{(l)} \times f^{(l)} \times n_C^{(l-1)} \times n_C^{(l)}" /></a> <br>
      Bias: <a href="https://www.codecogs.com/eqnedit.php?latex=1&space;\times&space;1&space;\times&space;1&space;\times&space;n_C^{(l)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?1&space;\times&space;1&space;\times&space;1&space;\times&space;n_C^{(l)}" title="1 \times 1 \times 1 \times n_C^{(l)}" /></a>

  - Activation: <a href="https://www.codecogs.com/eqnedit.php?latex=a^{(l)}&space;\rightarrow&space;n_H^{(l)}&space;\times&space;n_W^{(l)}&space;\times&space;n_C^{(l)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{(l)}&space;\rightarrow&space;n_H^{(l)}&space;\times&space;n_W^{(l)}&space;\times&space;n_C^{(l)}" title="a^{(l)} \rightarrow n_H^{(l)} \times n_W^{(l)} \times n_C^{(l)}" /></a>

      Activation vectorized: <a href="https://www.codecogs.com/eqnedit.php?latex=A^{(l)}&space;\rightarrow&space;m&space;\times&space;n_H^{(l)}&space;\times&space;n_W^{(l)}&space;\times&space;n_C^{(l)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A^{(l)}&space;\rightarrow&space;m&space;\times&space;n_H^{(l)}&space;\times&space;n_W^{(l)}&space;\times&space;n_C^{(l)}" title="A^{(l)} \rightarrow m \times n_H^{(l)} \times n_W^{(l)} \times n_C^{(l)}" /></a>

    <br><img src="Resources/deep_learning/cnn/convolution_layer.png" width=700>

### 1.6. Pooling layer

- Hyperparameters

  - <a href="https://www.codecogs.com/eqnedit.php?latex=f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f" title="f" /></a>: filter size
  - <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a>: stride size
  - Max or average pooling
  - Pooling usually does not use any padding

- Pooling

  - **Max-pooling layer:** slides an ( f, f ) window over the input and stores the max value of the window in the output.
  - **Average-pooling layer:** slides an ( f, f ) window over the input and stores the average value of the window in the output.

  <img src="Resources/deep_learning/cnn/pooling.png" width=350>

  <a href="https://www.codecogs.com/eqnedit.php?latex=n_H&space;\times&space;n_W&space;\times&space;n_C&space;\rightarrow&space;\left&space;\lfloor&space;\frac{n_H&space;-&space;f}{s}&plus;1&space;\right&space;\rfloor&space;\times&space;\left&space;\lfloor&space;\frac{n_H&space;-&space;f}{s}&plus;1&space;\right&space;\rfloor&space;\times&space;n_C" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_H&space;\times&space;n_W&space;\times&space;n_C&space;\rightarrow&space;\left&space;\lfloor&space;\frac{n_H&space;-&space;f}{s}&plus;1&space;\right&space;\rfloor&space;\times&space;\left&space;\lfloor&space;\frac{n_H&space;-&space;f}{s}&plus;1&space;\right&space;\rfloor&space;\times&space;n_C" title="n_H \times n_W \times n_C \rightarrow \left \lfloor \frac{n_H - f}{s}+1 \right \rfloor \times \left \lfloor \frac{n_H - f}{s}+1 \right \rfloor \times n_C" /></a>

  - Applies to each channels independently
  - No parameters for backpropagation to learn

- Advantages of pooling in ConvNet

  - (+) Reduces the size of the input
  - (+) Speeds up the computation
  - (+) Makes feature detectors more invariant to its position in the input

### 1.7. CNN schematics

- Types of layer in a convolutional network

  - Convolution (CONV)
  - Pooling (POOL)
  - Fully-connected (FC)

- Workflow

  <img src="Resources/deep_learning/cnn/cnn_example.png" width=700>

  <img src="Resources/deep_learning/cnn/cnn_example_table.png" width=450>

  <br>

  From left to right, the height and width often decrease, and the number of channels often increase.

### 1.8. Why convolutions

- Advantages of convolutional layers over fully connected layers

  - (+) **Parameter sharing:** A feature detector (such as a vertical edge detector) that’s useful in one part of the image is probably useful in another part of the image.

    - Reduces the total number of parameters, thus reducing overfitting.
    - Allows a feature detector to be used in multiple locations throughout the whole input image/input volume. Good at capturing **translation invariance** (e.g. over different places in a picture) because you are applying the same filter.

  - (+) **Sparsity of connections:** In each layer, each output value depends only on a small number of inputs.

- Putting all together

  <img src="Resources/deep_learning/cnn/cnn_example2.png" width=500>

- Learn more about convolutional neural networks

  - [An intro to ConvNet and Image Recognition on YouTube](https://www.youtube.com/watch?v=2-Ol7ZB0MmU)
  - [An intuitive guide to ConvNet on Medium](https://medium.freecodecamp.org/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050)

## 2. CNN case studies

### 2.1. Classic networks

- #### LeNet-5

    <img src="Resources/deep_learning/cnn/lenet5.png" width=700> <br>
    [LeCun et al., 1998. Gradient-based learning applied to document recognition]

    - The goal of LeNet-5 was to recognize handwritten digits on grayscale images of 32 by 32 by 1.
    - From left to right, the height and width decrease, and the number of channels increases.
    - ~60K parameters

- #### AlexNet

    <img src="Resources/deep_learning/cnn/alexnet.png" width=700> <br>
    [Krizhevsky et al., 2012. ImageNet classification with deep convolutional neural networks]

    - Similar to LeNet-5 but much bigger
    - Uses ReLu activation rather than sigmoid/tanh
    - ~60M parameters

- #### VGG

    <img src="Resources/deep_learning/cnn/vgg.png" width=700> <br>
    [Simonyan & Zisserman 2015. Very deep convolutional networks for large-scale image recognition]

    - Relatively deep CNN, 16 layers that have weights
    - Simplified architecture, with same hyperparameters among CONV layers, and same hyperparameters among POOL layers, except for the number of channels that roughly doubles for every step.
    - ~138M parameters

### 2.2. ResNet

- #### Residual block

    <a href="https://www.codecogs.com/eqnedit.php?latex=a^{(l&plus;2)}&space;=&space;g(z^{(l&plus;2)}&plus;a^{(l)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{(l&plus;2)}&space;=&space;g(z^{(l&plus;2)}&plus;a^{(l)})" title="a^{(l+2)} = g(z^{(l+2)}+a^{(l)})" /></a>

    <img src="Resources/deep_learning/cnn/residual_block.png" width=350> <br>
    [He et al., 2015. Deep residual networks for image recognition]

    - <a href="https://www.codecogs.com/eqnedit.php?latex=z^{(l&plus;2)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z^{(l&plus;2)}" title="z^{(l+2)}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=a^{(l)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{(l)}" title="a^{(l)}" /></a> usually have the same dimension by using "same" convolution. When they don't have the same dimension, implement an additional matrix to convert <a href="https://www.codecogs.com/eqnedit.php?latex=a^{(l)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{(l)}" title="a^{(l)}" /></a>'s dimension to <a href="https://www.codecogs.com/eqnedit.php?latex=z^{(l&plus;2)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z^{(l&plus;2)}" title="z^{(l+2)}" /></a>'s. <a href="https://www.codecogs.com/eqnedit.php?latex=a^{(l&plus;2)}&space;=&space;g(z^{(l&plus;2)}&plus;w_sa^{(l)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{(l&plus;2)}&space;=&space;g(z^{(l&plus;2)}&plus;w_sa^{(l)})" title="a^{(l+2)} = g(z^{(l+2)}+w_sa^{(l)})" /></a>

- #### Residual network

    <img src="Resources/deep_learning/cnn/residual_network.png" width=600>

    <img src="Resources/deep_learning/cnn/resnet_vs_plain.png" width=500>

    - (+) Helps with the vanishing and exploding gradient problems
    - (+) Allows to train much deeper neural networks without really appreciable loss in performance

- #### Why ResNets work

    - (+) Never hurt performance: Adding two layers in the neural network doesn't hurt the network's ability to do as well as this simpler network without these two extra layers, because identity function is easy for residual block to learn, which is not so easy in plain network.

### 2.3. Inception neural networks

- #### Networks in networks and 1 x 1 convolutions

    <img src="Resources/deep_learning/cnn/networks_in_networks.png" width=500> <br>
    [Lin et al., 2013. Network in network]

    - (+) Allows to shrink the number of channels and therefore, saves computation in some networks
    - (+) Adds non-linearity by allowings to learn the more complex function

- #### Inception network motivation

    <img src="Resources/deep_learning/cnn/inception_network_motivation.png" width=500> <br>
    [Szegedy et al. 2014. Going deeper with convolutions]

- #### Use 1 x 1 convolutions to solve the computational cost problem

    - Convolution can be very costly

        <img src="Resources/deep_learning/cnn/inception_computation_cost.png" width=350>

    - Use 1 x 1 convolution to reduce cost by a factor of ~10

        <img src="Resources/deep_learning/cnn/inception_computation_cost2.png" width=500>

- #### Inception module

    <img src="Resources/deep_learning/cnn/inception_module.png" width=400> <br>
    Building block of the inception network <br>
    [Szegedy et al., 2014, Going Deeper with Convolutions]

- #### Inception network

    <img src="Resources/deep_learning/cnn/inception_network.png" width=700> <br>
    This particular network inception network is also known as "GoogleNet". <br>
    [Szegedy et al., 2014, Going Deeper with Convolutions]

    - (+) The additional outputs from the hidden layers help to ensure that the features computed even in the hidden layers not too bad for predicting the output class of a image, which appears to have a regularizing effect on the inception network and helps prevent this network from overfitting.

## 3. Practical advices for using CNN

### 3.1. General tips

- Use open source implementation instead of re-inventing the wheels

  - Use architectures of networks published in the literature
  - Use open source implementations if possible

- Transfer learning

  - Use pretrained models and fine tune on your dataset

- Data augmentation to improve the performance of computer vision systems

### 3.2. Data vs. hand-engineering

- 2 sources of knowledge

    - Labeled data
    - Hand-engineered features/network architecture/other components

    <img src="Resources/deep_learning/cnn/data_hand_engineering.png" width=600>

### 3.3. Tips for doing well on benchmarks/winning competitions

- Warning: not necessarily practical and rarely used when building production systems
- Ensembling: train several networks independently and average their outputs
- Multi-crop at test time: run classifier on multiple versions of test images and average results

