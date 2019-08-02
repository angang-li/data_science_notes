# Convolutional Neural Networks

- [Convolutional Neural Networks](#convolutional-neural-networks)
  - [1. Foundations of convolutional neural networks](#1-foundations-of-convolutional-neural-networks)
    - [1.1. Computer vision](#11-computer-vision)
    - [1.2. Convolution operation](#12-convolution-operation)
    - [1.3. Edge detection](#13-edge-detection)
    - [1.4. Padding](#14-padding)
    - [1.5. Strided convolution](#15-strided-convolution)
    - [1.6. Convolution layer](#16-convolution-layer)
    - [1.7. Pooling layer](#17-pooling-layer)
    - [1.8. CNN schematics](#18-cnn-schematics)
    - [1.9. Why convolutions](#19-why-convolutions)

## 1. Foundations of convolutional neural networks

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

### 1.3. Edge detection

- Edge example

  <img src="Resources/deep_learning/cnn/edge_example.png" width=400>

- Vertical edge detection

  <img src="Resources/deep_learning/cnn/vertical_edge_detection.png" width=400>

- Hand-coded edge detection filters

  <img src="Resources/deep_learning/cnn/edge_detection_filters.png" width=400>

  Dimension: <a href="https://www.codecogs.com/eqnedit.php?latex=f&space;\times&space;f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f&space;\times&space;f" title="f \times f" /></a>, where f is usually odd

- Learn the filter parameters from neural network

  <img src="Resources/deep_learning/cnn/edge_detection_parameterize.png" width=150>

### 1.4. Padding

- Downside of convolution operation

  - (-) Shrinking output
  - (-) throwing away information from edges

- Padding

  Common to zero-pad the border. In the example below, the padding p = 1.

  <img src="Resources/deep_learning/cnn/keras_conv2d_padding.gif" width=450>

- Valid and same convolutions

  - Valid convolution: no padding

    <a href="https://www.codecogs.com/eqnedit.php?latex=n\times&space;n&space;\&space;image&space;\&space;\&space;\ast&space;\&space;\&space;f&space;\times&space;f&space;\&space;filter&space;\&space;\&space;\rightarrow&space;\&space;\&space;(n-f&plus;1)&space;\times&space;(n-f&plus;1)&space;\&space;output" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n\times&space;n&space;\&space;image&space;\&space;\&space;\ast&space;\&space;\&space;f&space;\times&space;f&space;\&space;filter&space;\&space;\&space;\rightarrow&space;\&space;\&space;(n-f&plus;1)&space;\times&space;(n-f&plus;1)&space;\&space;output" title="n\times n \ image \ \ \ast \ \ f \times f \ filter \ \ \rightarrow \ \ (n-f+1) \times (n-f+1) \ output" /></a>

  - Same convolution: pad so that output size is the same as the input sizes

    <a href="https://www.codecogs.com/eqnedit.php?latex=p=\frac{f-1}{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p=\frac{f-1}{2}" title="p=\frac{f-1}{2}" /></a>

- Advantages of padding

  - (+) Allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. This is important for building deeper networks, since otherwise the height/width would shrink as you go to deeper layers.
  - (+) Helps us keep more of the information at the border of an image. Without padding, very few values at the next layer would be affected by pixels as the edges of an image.

### 1.5. Strided convolution

- Stride example

  In the example below, stride s = 2.

  <img src="Resources/deep_learning/cnn/stride.png" width=450>

  Output dimension: <a href="https://www.codecogs.com/eqnedit.php?latex=\left&space;\lfloor&space;\frac{n&plus;2p-f}{s}&plus;1&space;\right&space;\rfloor&space;\&space;\&space;\times&space;\&space;\&space;\left&space;\lfloor&space;\frac{n&plus;2p-f}{s}&plus;1&space;\right&space;\rfloor" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left&space;\lfloor&space;\frac{n&plus;2p-f}{s}&plus;1&space;\right&space;\rfloor&space;\&space;\&space;\times&space;\&space;\&space;\left&space;\lfloor&space;\frac{n&plus;2p-f}{s}&plus;1&space;\right&space;\rfloor" title="\left \lfloor \frac{n+2p-f}{s}+1 \right \rfloor \ \ \times \ \ \left \lfloor \frac{n+2p-f}{s}+1 \right \rfloor" /></a> <br>
  By convention, the filter must be fully contained in the input image to do convolution.

### 1.6. Convolution layer

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

### 1.7. Pooling layer

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

- Pdvantages of pooling in ConvNet

  - (+) Reduces the size of the input
  - (+) Speeds up the computation
  - (+) Makes feature detectors more invariant to its position in the input

### 1.8. CNN schematics

- Types of layer in a convolutional network

  - Convolution (CONV)
  - Pooling (POOL)
  - Fully-connected (FC)

- Schematics

  <img src="Resources/deep_learning/cnn/cnn_example.png" width=700>

  <img src="Resources/deep_learning/cnn/cnn_example_table.png" width=450>

  <br>

  From left to right, the height and width often decrease, and the number of channels often increase.

### 1.9. Why convolutions

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
