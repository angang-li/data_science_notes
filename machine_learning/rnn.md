# Sequence Models

- [Sequence Models](#sequence-models)
  - [1. Recurrent neural networks (RNN)](#1-recurrent-neural-networks-rnn)
    - [1.1. Motivation](#11-motivation)
    - [1.2. RNN model structure](#12-rnn-model-structure)
    - [1.3. Different types of RNNs](#13-different-types-of-rnns)
    - [1.4. Language model and sequence generation](#14-language-model-and-sequence-generation)
    - [1.5. GRU and LSTM to resolve vanishing gradients of RNN](#15-gru-and-lstm-to-resolve-vanishing-gradients-of-rnn)
    - [1.6. BRNN and deep RNN to build more powerful models](#16-brnn-and-deep-rnn-to-build-more-powerful-models)

## 1. Recurrent neural networks (RNN)

### 1.1. Motivation

- Examples of sequence data

  <img src="Resources/deep_learning/rnn/sequence_examples.png" width=600>

- Notation

  - Motivating example

    x: Harry Potter and Hermione Granger invented a new spell. <br>
    y: 1, 1, 0, 1, 1, 0, 0, 0, 0

    - <a href="https://www.codecogs.com/eqnedit.php?latex=x^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="x^{\left \langle t \right \rangle}" /></a>: t-th element in the sequence of training example
    - <a href="https://www.codecogs.com/eqnedit.php?latex=T_x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_x" title="T_x" /></a>: the length of the input sequence, variable across different training examples
    - <a href="https://www.codecogs.com/eqnedit.php?latex=T_y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_y" title="T_y" /></a>: the length of the output sequence, not necessarily the same as <a href="https://www.codecogs.com/eqnedit.php?latex=T_x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_x" title="T_x" /></a>

  - Representing words

    - Represent <a href="https://www.codecogs.com/eqnedit.php?latex=x^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="x^{\left \langle t \right \rangle}" /></a> by a one-hot vector that spans the entire list of vocabulary
    - Use `<UNK>` (unknown word) to represent words not in the vocabulary
    - Represent sentence ends by adding an extra token called `<EOS>` (end of sentence)

### 1.2. RNN model structure

- Problems of a standard network

  <img src="Resources/deep_learning/rnn/standard_nn.png" width=400>

  - (-) Inputs, outputs can be different lengths in different examples.
  - (-) Doesn’t share features learned across different positions of text.

- RNN structure

    The recurrent neural network scans through the data from left to right. The parameters it uses for each time step are shared.

  - When <a href="https://www.codecogs.com/eqnedit.php?latex=T_x=T_y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_x=T_y" title="T_x=T_y" /></a>

    <img src="Resources/deep_learning/rnn/rnn.png" width=400>, or equivalently, <img src="Resources/deep_learning/rnn/rnn_equivalent.png" width=120>

  - (-) Only uses the information that is earlier in the sequence to make a prediction. E.g., when predicting y2, it doesn't use information about the words X3, X4, X5 and so on.
    - Can be resolved using Bidirectional RNN (BRNN)

- Forward propogation

  - By convention, initialize <a href="https://www.codecogs.com/eqnedit.php?latex=a^{\left&space;\langle&space;0&space;\right&space;\rangle}=\vec{0}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{\left&space;\langle&space;0&space;\right&space;\rangle}=\vec{0}" title="a^{\left \langle 0 \right \rangle}=\vec{0}" /></a>

  - Activation
  
    <a href="https://www.codecogs.com/eqnedit.php?latex=a^{\left&space;\langle&space;t&space;\right&space;\rangle}=g(W_{aa}&space;a^{\left&space;\langle&space;t-1&space;\right&space;\rangle}&plus;W_{ax}&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}&plus;b_a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{\left&space;\langle&space;t&space;\right&space;\rangle}=g(W_{aa}&space;a^{\left&space;\langle&space;t-1&space;\right&space;\rangle}&plus;W_{ax}&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}&plus;b_a)" title="a^{\left \langle t \right \rangle}=g(W_{aa} a^{\left \langle t-1 \right \rangle}+W_{ax} x^{\left \langle t \right \rangle}+b_a)" /></a>, where the activation function is usually `tanh` or `ReLU`

  - Output
  
    <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle}=g(W_{ya}&space;a^{\left&space;\langle&space;t&space;\right&space;\rangle}&plus;b_y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle}=g(W_{ya}&space;a^{\left&space;\langle&space;t&space;\right&space;\rangle}&plus;b_y)" title="\hat{y}^{\left \langle t \right \rangle}=g(W_{ya} a^{\left \langle t \right \rangle}+b_y)" /></a>, where the activation function is usually sigmoid

- Simplified RNN notation

  - Activation: <a href="https://www.codecogs.com/eqnedit.php?latex=a^{\left&space;\langle&space;t&space;\right&space;\rangle}=g(W_{a}&space;\left&space;[a^{\left&space;\langle&space;t-1&space;\right&space;\rangle},&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}&space;\right&space;]&plus;b_a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{\left&space;\langle&space;t&space;\right&space;\rangle}=g(W_{a}&space;\left&space;[a^{\left&space;\langle&space;t-1&space;\right&space;\rangle},&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}&space;\right&space;]&plus;b_a)" title="a^{\left \langle t \right \rangle}=g(W_{a} \left [a^{\left \langle t-1 \right \rangle}, x^{\left \langle t \right \rangle} \right ]+b_a)" /></a>, where <a href="https://www.codecogs.com/eqnedit.php?latex=W_{a}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W_{a}" title="W_{a}" /></a> stacks <a href="https://www.codecogs.com/eqnedit.php?latex=W_{aa}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W_{aa}" title="W_{aa}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=W_{ax}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W_{ax}" title="W_{ax}" /></a>
  - Output: <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle}=g(W_{y}&space;a^{\left&space;\langle&space;t&space;\right&space;\rangle}&plus;b_y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle}=g(W_{y}&space;a^{\left&space;\langle&space;t&space;\right&space;\rangle}&plus;b_y)" title="\hat{y}^{\left \langle t \right \rangle}=g(W_{y} a^{\left \langle t \right \rangle}+b_y)" /></a>

- Loss function

  - Loss at a single prediction at a single time step

    <a href="https://www.codecogs.com/eqnedit.php?latex=E^{\left&space;\langle&space;t&space;\right&space;\rangle}(\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle},&space;y^{\left&space;\langle&space;t&space;\right&space;\rangle})&space;=&space;-y^{\left&space;\langle&space;t&space;\right&space;\rangle}&space;log\&space;\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle}&space;-&space;(1-y^{\left&space;\langle&space;t&space;\right&space;\rangle})&space;log(1-\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E^{\left&space;\langle&space;t&space;\right&space;\rangle}(\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle},&space;y^{\left&space;\langle&space;t&space;\right&space;\rangle})&space;=&space;-y^{\left&space;\langle&space;t&space;\right&space;\rangle}&space;log\&space;\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle}&space;-&space;(1-y^{\left&space;\langle&space;t&space;\right&space;\rangle})&space;log(1-\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle})" title="E^{\left \langle t \right \rangle}(\hat{y}^{\left \langle t \right \rangle}, y^{\left \langle t \right \rangle}) = -y^{\left \langle t \right \rangle} log\ \hat{y}^{\left \langle t \right \rangle} - (1-y^{\left \langle t \right \rangle}) log(1-\hat{y}^{\left \langle t \right \rangle})" /></a>

  - Overall loss of the entire sequence

    <a href="https://www.codecogs.com/eqnedit.php?latex=J(\hat{y},&space;y)&space;=&space;\sum_{t=1}^{T_y}&space;E^{\left&space;\langle&space;t&space;\right&space;\rangle}(\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle},&space;y^{\left&space;\langle&space;t&space;\right&space;\rangle})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(\hat{y},&space;y)&space;=&space;\sum_{t=1}^{T_y}&space;E^{\left&space;\langle&space;t&space;\right&space;\rangle}(\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle},&space;y^{\left&space;\langle&space;t&space;\right&space;\rangle})" title="J(\hat{y}, y) = \sum_{t=1}^{T_y} E^{\left \langle t \right \rangle}(\hat{y}^{\left \langle t \right \rangle}, y^{\left \langle t \right \rangle})" /></a>

- Backpropagation through time

  - Overall backprop

    <img src="Resources/deep_learning/rnn/backprop_thru_time.png" width=400>

  - RNN cell backprop

    <img src="Resources/deep_learning/rnn/rnn_cell_backprop.png" width=750>

### 1.3. Different types of RNNs

- #### Many-to-many, Tx = Ty

  <a href="https://www.codecogs.com/eqnedit.php?latex=T_x=T_y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_x=T_y" title="T_x=T_y" /></a>

  <img src="Resources/deep_learning/rnn/rnn_m2m.png" width=300>

- #### Many-to-one

  x = text, y = 0/1 or 1,...,5

  <img src="Resources/deep_learning/rnn/rnn_m21.png" width=300> e.g. sentiment classification

- #### One-to-many

  <a href="https://www.codecogs.com/eqnedit.php?latex=x&space;\rightarrow&space;y^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;y^{\left&space;\langle&space;2&space;\right&space;\rangle},&space;...,&space;y^{\left&space;\langle&space;T_y&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x&space;\rightarrow&space;y^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;y^{\left&space;\langle&space;2&space;\right&space;\rangle},&space;...,&space;y^{\left&space;\langle&space;T_y&space;\right&space;\rangle}" title="x \rightarrow y^{\left \langle 1 \right \rangle}, y^{\left \langle 2 \right \rangle}, ..., y^{\left \langle T_y \right \rangle}" /></a>

  <img src="Resources/deep_learning/rnn/rnn_12m.png" width=310> e.g. music generation

- #### Many-to-many, Tx ≠ Ty

  <a href="https://www.codecogs.com/eqnedit.php?latex=T_x&space;\neq&space;T_y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_x&space;\neq&space;T_y" title="T_x \neq T_y" /></a>

  <img src="Resources/deep_learning/rnn/rnn_m2m_general.png" width=430> e.g. machine translation

### 1.4. Language model and sequence generation

- Tokenize input text

  Training set: large corpus of English text

  - E.g. y = Cats average 15 hours of sleep a day. \<EOS> (end of sentence)
  - E.g. The Egyptian \<UNK> (unknown word) is a bread of cat. \<EOS>

- Model training

  <a href="https://www.codecogs.com/eqnedit.php?latex=x^{\left&space;\langle&space;t&space;\right&space;\rangle}&space;=&space;y^{\left&space;\langle&space;t-1&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x^{\left&space;\langle&space;t&space;\right&space;\rangle}&space;=&space;y^{\left&space;\langle&space;t-1&space;\right&space;\rangle}" title="x^{\left \langle t \right \rangle} = y^{\left \langle t-1 \right \rangle}" /></a>

  Each step in the RNN looks at some set of preceding words, and given these words, what is the probability of the next word. The RNN learns to predict one word at a time going from left to right.

  <img src="Resources/deep_learning/rnn/language_modeling.png" width=500>

  - Activation

    <a href="https://www.codecogs.com/eqnedit.php?latex=a^{\langle&space;t&plus;1&space;\rangle}&space;=&space;\tanh(W_{ax}&space;x^{\langle&space;t&plus;1&space;\rangle&space;}&space;&plus;&space;W_{aa}&space;a^{\langle&space;t&space;\rangle&space;}&space;&plus;&space;b)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{\langle&space;t&plus;1&space;\rangle}&space;=&space;\tanh(W_{ax}&space;x^{\langle&space;t&plus;1&space;\rangle&space;}&space;&plus;&space;W_{aa}&space;a^{\langle&space;t&space;\rangle&space;}&space;&plus;&space;b)" title="a^{\langle t+1 \rangle} = \tanh(W_{ax} x^{\langle t+1 \rangle } + W_{aa} a^{\langle t \rangle } + b)" /></a>

  - Output

    <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}^{\langle&space;t&plus;1&space;\rangle&space;}&space;=&space;softmax(W_{ya}&space;a^{\langle&space;t&space;&plus;&space;1&space;\rangle&space;}&space;&plus;&space;b_y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}^{\langle&space;t&plus;1&space;\rangle&space;}&space;=&space;softmax(W_{ya}&space;a^{\langle&space;t&space;&plus;&space;1&space;\rangle&space;}&space;&plus;&space;b_y)" title="\hat{y}^{\langle t+1 \rangle } = softmax(W_{ya} a^{\langle t + 1 \rangle } + b_y)" /></a>

  - Individual error

    <a href="https://www.codecogs.com/eqnedit.php?latex=E^{\left&space;\langle&space;t&space;\right&space;\rangle}(\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle},&space;y^{\left&space;\langle&space;t&space;\right&space;\rangle})&space;=&space;-\sum_i&space;y_i^{\left&space;\langle&space;t&space;\right&space;\rangle}&space;log\&space;\hat{y}_i^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E^{\left&space;\langle&space;t&space;\right&space;\rangle}(\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle},&space;y^{\left&space;\langle&space;t&space;\right&space;\rangle})&space;=&space;-\sum_i&space;y_i^{\left&space;\langle&space;t&space;\right&space;\rangle}&space;log\&space;\hat{y}_i^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="E^{\left \langle t \right \rangle}(\hat{y}^{\left \langle t \right \rangle}, y^{\left \langle t \right \rangle}) = -\sum_i y_i^{\left \langle t \right \rangle} log\ \hat{y}_i^{\left \langle t \right \rangle}" /></a>

  - Overall cost function

    <a href="https://www.codecogs.com/eqnedit.php?latex=J&space;=&space;\sum_t&space;E^{\left&space;\langle&space;t&space;\right&space;\rangle}(\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle},&space;y^{\left&space;\langle&space;t&space;\right&space;\rangle})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J&space;=&space;\sum_t&space;E^{\left&space;\langle&space;t&space;\right&space;\rangle}(\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle},&space;y^{\left&space;\langle&space;t&space;\right&space;\rangle})" title="J = \sum_t E^{\left \langle t \right \rangle}(\hat{y}^{\left \langle t \right \rangle}, y^{\left \langle t \right \rangle})" /></a>

- Sampling novel sequences

  To generate a randomly chosen sentence from the RNN language model, randomly sample according to this softmax distribution at each time step.

  <img src="Resources/deep_learning/rnn/language_modeling_sample.png" width=445>

- Character level vs. word level language model

  Word level vocabulary = [a, aaron, ..., zulu, \<UNK>] <br>
  Character level vocabulary = [a, ..., z, 0, ..., 9, A, ..., Z, ., ..., ;]

  - (+) Character level language model doesn't have to worry about unknown word.
  - (-) Character level language model ends up with much longer sequences. Not as good as word level language models at capturing long range dependencies between how the the earlier parts of the sentence also affect the later part of the sentence.
  - (-) Character level language model is more computationally expensive to train.

### 1.5. GRU and LSTM to resolve vanishing gradients of RNN

- #### Weaknesses of basic RNN

  - (-) Vanishing gradient problem: Language can have very long-term dependencies, where a word much earlier can affect what needs to come much later in the sentence. But the basic RNN is not very good at capturing very long-term dependencies, given that the model has many local influences.
    - Hard to resolve.

  - (-) Exploding gradient problem: exponentially large gradients can cause the parameters to blow up.
    - Easy to observe: `NaN`.
    - One solution is to apply gradient clipping, i.e., when the gradient vector is bigger than some threshold, re-scale some of the gradient vectors so that it is not too big.

- #### Gated Recurrent Unit (GRU)

  (+) Captures long range connections and resolves the vanishing gradient problems <br>
  (+) Simpler model than LSTM, more computationally efficient

  - Notation

    - c = memory cell, determines whether the subject of the sentence is e.g. singular or plural
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{c}^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{c}^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="\tilde{c}^{\left \langle t \right \rangle}" /></a> = candidate to replace <a href="https://www.codecogs.com/eqnedit.php?latex=c^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="c^{\left \langle t \right \rangle}" /></a>
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\Gamma_r" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Gamma_r" title="\Gamma_r" /></a> = relevance, how relevant is <a href="https://www.codecogs.com/eqnedit.php?latex=c^{\left&space;\langle&space;t-1&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c^{\left&space;\langle&space;t-1&space;\right&space;\rangle}" title="c^{\left \langle t-1 \right \rangle}" /></a> to computing the next candidate <a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{c}^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{c}^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="\tilde{c}^{\left \langle t \right \rangle}" /></a>
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\Gamma_u" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Gamma_u" title="\Gamma_u" /></a> = gate, where u stands for update gate. Determines whether to update c

  - RNN unit

    <img src="Resources/deep_learning/rnn/rnn_unit.png" width=500>

  - GRU unit

    <img src="Resources/deep_learning/rnn/gru_unit.png" width=400>

    [Cho et al., 2014. On the properties of neural machine translation: Encoder-decoder approaches], [Chung et al., 2014. Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling]

  - Steps

    - At every time-step, Initialize the memory cell

      <a href="https://www.codecogs.com/eqnedit.php?latex=c^{\left&space;\langle&space;t-1&space;\right&space;\rangle}=a^{\left&space;\langle&space;t-1&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c^{\left&space;\langle&space;t-1&space;\right&space;\rangle}=a^{\left&space;\langle&space;t-1&space;\right&space;\rangle}" title="c^{\left \langle t-1 \right \rangle}=a^{\left \langle t-1 \right \rangle}" /></a>

    - Calculate the candidate new value <a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{c}^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{c}^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="\tilde{c}^{\left \langle t \right \rangle}" /></a> for the memory cell

      <a href="https://www.codecogs.com/eqnedit.php?latex=\Gamma_r=\sigma(W_r&space;[c^{\left&space;\langle&space;t-1&space;\right&space;\rangle},&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}]&space;&plus;&space;b_r)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Gamma_r=\sigma(W_r&space;[c^{\left&space;\langle&space;t-1&space;\right&space;\rangle},&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}]&space;&plus;&space;b_r)" title="\Gamma_r=\sigma(W_r [c^{\left \langle t-1 \right \rangle}, x^{\left \langle t \right \rangle}] + b_r)" /></a>

      <a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{c}^{\left&space;\langle&space;t&space;\right&space;\rangle}=tanh(W_c&space;[\Gamma_r*c^{\left&space;\langle&space;t-1&space;\right&space;\rangle},&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}]&space;&plus;&space;b_c)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{c}^{\left&space;\langle&space;t&space;\right&space;\rangle}=tanh(W_c&space;[\Gamma_r*c^{\left&space;\langle&space;t-1&space;\right&space;\rangle},&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}]&space;&plus;&space;b_c)" title="\tilde{c}^{\left \langle t \right \rangle}=tanh(W_c [\Gamma_r*c^{\left \langle t-1 \right \rangle}, x^{\left \langle t \right \rangle}] + b_c)" /></a>, where * represents element-wise multiplication

    - Calculate gate to decide whether to update the memory cell

      <a href="https://www.codecogs.com/eqnedit.php?latex=\Gamma_u=\sigma(W_u&space;[c^{\left&space;\langle&space;t-1&space;\right&space;\rangle},&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}]&space;&plus;&space;b_u)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Gamma_u=\sigma(W_u&space;[c^{\left&space;\langle&space;t-1&space;\right&space;\rangle},&space;x^{\left&space;\langle&space;t&space;\right&space;\rangle}]&space;&plus;&space;b_u)" title="\Gamma_u=\sigma(W_u [c^{\left \langle t-1 \right \rangle}, x^{\left \langle t \right \rangle}] + b_u)" /></a>

      Gate is very good at maintaining the value for the cell. Because gate can be very close to zero and one, it doesn't suffer much from vanishing gradient problem.

    - Update the memory cell

      <a href="https://www.codecogs.com/eqnedit.php?latex=c^{\left&space;\langle&space;t&space;\right&space;\rangle}=\Gamma_u&space;*&space;\tilde{c}^{\left&space;\langle&space;t&space;\right&space;\rangle}&space;&plus;&space;(1-\Gamma_u)&space;*&space;c^{\left&space;\langle&space;t-1&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c^{\left&space;\langle&space;t&space;\right&space;\rangle}=\Gamma_u&space;*&space;\tilde{c}^{\left&space;\langle&space;t&space;\right&space;\rangle}&space;&plus;&space;(1-\Gamma_u)&space;*&space;c^{\left&space;\langle&space;t-1&space;\right&space;\rangle}" title="c^{\left \langle t \right \rangle}=\Gamma_u * \tilde{c}^{\left \langle t \right \rangle} + (1-\Gamma_u) * c^{\left \langle t-1 \right \rangle}" /></a>, where * represents element-wise multiplication

    - Equalize hidden state a and memory cell c

      <a href="https://www.codecogs.com/eqnedit.php?latex=a^{\left&space;\langle&space;t&space;\right&space;\rangle}=c^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{\left&space;\langle&space;t&space;\right&space;\rangle}=c^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="a^{\left \langle t \right \rangle}=c^{\left \langle t \right \rangle}" /></a>

- #### Long Short Term Memory (LSTM)

  (+) Can be more powerful than GRU to learn very long range connections in a sequence

  - Notation

    - <a href="https://www.codecogs.com/eqnedit.php?latex=\Gamma_u" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Gamma_u" title="\Gamma_u" /></a> = update gate
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\Gamma_f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Gamma_f" title="\Gamma_f" /></a> = forget gate
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\Gamma_o" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Gamma_o" title="\Gamma_o" /></a> = output gate
    - <a href="https://www.codecogs.com/eqnedit.php?latex=c^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="c^{\left \langle t \right \rangle}" /></a> is different from <a href="https://www.codecogs.com/eqnedit.php?latex=a^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="a^{\left \langle t \right \rangle}" /></a>

  - LSTM vs. GRU steps

    <img src="Resources/deep_learning/rnn/lstm_gru_eq.png" width=600>
    , where * represents element-wise multiplication

  - LSTM unit

    <img src="Resources/deep_learning/rnn/lstm_unit.png" width=500>

    [Hochreiter & Schmidhuber 1997. Long short-term memory]

### 1.6. BRNN and deep RNN to build more powerful models

- #### Bidirectional RNN (BRNN)

  - BRNN model structure

    <img src="Resources/deep_learning/rnn/brnn.png" width=500>

    <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle}&space;=&space;g(W_y&space;[\overrightarrow{a}^{\left&space;\langle&space;t&space;\right&space;\rangle},&space;\overleftarrow{a}^{\left&space;\langle&space;t&space;\right&space;\rangle}]&plus;b_y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle}&space;=&space;g(W_y&space;[\overrightarrow{a}^{\left&space;\langle&space;t&space;\right&space;\rangle},&space;\overleftarrow{a}^{\left&space;\langle&space;t&space;\right&space;\rangle}]&plus;b_y)" title="\hat{y}^{\left \langle t \right \rangle} = g(W_y [\overrightarrow{a}^{\left \langle t \right \rangle}, \overleftarrow{a}^{\left \langle t \right \rangle}]+b_y)" /></a>

    BRNN commonly used with LSTM units, although is compatible with either RNN, GRU, or LSTM units.

  - Pros and cons

    - (+) Anywhere in the sequence, can make predictions using information from both earlier and later in the sequence.
    - (-) Need the entire sequence of data before making predictions anywhere.
      - The standard BRNN not effective for real-time speech recognition.
      - Very effective for lots of NLP applications where the entire sentence is available all at the same time.

- #### Deep RNNs

  - Example of deep RNN with 3 hidden layers

    <img src="Resources/deep_learning/rnn/deep_rnn.png" width=550>

    RNNs generally don’t have a lot of hidden layers (3 is a lot), because of the temporal dimensions.
