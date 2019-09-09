# Sequence Models

- [Sequence Models](#sequence-models)
  - [1. Recurrent neural networks (RNN)](#1-recurrent-neural-networks-rnn)
    - [1.1. Motivation](#11-motivation)
    - [1.2. RNN model structure](#12-rnn-model-structure)
    - [1.3. Different types of RNNs](#13-different-types-of-rnns)
    - [1.4. Language model and sequence generation](#14-language-model-and-sequence-generation)
    - [1.5. GRU and LSTM to resolve vanishing gradients of RNN](#15-gru-and-lstm-to-resolve-vanishing-gradients-of-rnn)
    - [1.6. BRNN and deep RNN to build more powerful models](#16-brnn-and-deep-rnn-to-build-more-powerful-models)
  - [2. Natural language processing & word embeddings](#2-natural-language-processing--word-embeddings)
    - [2.1. Intro to word embeddings](#21-intro-to-word-embeddings)
    - [2.2. Learning word embeddings: Word2vec and GloVe](#22-learning-word-embeddings-word2vec-and-glove)
    - [2.3. Applications using word embeddings](#23-applications-using-word-embeddings)
  - [3. Sequence models & attention mechanism](#3-sequence-models--attention-mechanism)
    - [3.1. Various sequence to sequence architectures](#31-various-sequence-to-sequence-architectures)
    - [3.2. Beam search algorithm](#32-beam-search-algorithm)
    - [3.3. Bleu score](#33-bleu-score)
    - [3.4. Attention model](#34-attention-model)
    - [3.5. Speech recognition - audio data](#35-speech-recognition---audio-data)

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

## 2. Natural language processing & word embeddings

### 2.1. Intro to word embeddings

- #### Word representation

  - One-hot representation

    E.g. V = [a, aaron, ..., zulu, \<UNK>]

    Use <a href="https://www.codecogs.com/eqnedit.php?latex=O_{5391}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?O_{5391}" title="O_{5391}" /></a> to represent the one-hot vector of the 5391st word in the vocabulary.

    - (-) Treats each word as a thing by itself, and does not allow an algorithm to easily generalize across words

  - Featurized representation: word embedding

    Each word can be represented by a high-dimensional vector of features (e.g. gender, royal, age, food, cost, alive, verb, etc.). Use <a href="https://www.codecogs.com/eqnedit.php?latex=e_{5391}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e_{5391}" title="e_{5391}" /></a> to represent the featurized vector of the 5391st word in the vocabulary.

    Featurization from an embedding algorithm is not necessarily interpretable.

  - Visualizing word embeddings

    t-SNE maps high dimensional data (e.g. 300D) into a 2D space. The mapping is highly non-linear.

    <img src="Resources/deep_learning/rnn/t_sne.png" width=350>

    [van der Maaten and Hinton., 2008. Visualizing data using t-SNE]

- #### Using word embeddings with tranfer learning

  1. Learn word embeddings from large text corpus. (1-100B words). (Or download pre-trained embedding online.)  

  2. Transfer embedding to new task with smaller training set. (say, 100k words)

      - (+) Can now use relatively lower dimensional feature vectors compared to one-hot vector

  3. Optional: Continue to fine tune the word embeddings with new data, if training set is relatively large.

- #### Relations to face encoding

  - Word embedding learns a fixed encoding (vector e) for each of the words in the vocabulary, whereas face recognition can encode unseen images.
  - "Encoding" and "embedding" are used somewhat interchangeably.

- #### Properties of word embeddings

  - Analogies using word vectors

    (+) Word embedding can help with analogy reasoning. Research reported 30% to 75% accuracy on analogy.

    <img src="Resources/deep_learning/rnn/t_sne_analogy.png" width=450>

    [Mikolov et. al., 2013, Linguistic regularities in continuous space word representations]

  - Cosine similarity

    <a href="https://www.codecogs.com/eqnedit.php?latex=sim(u,&space;v)&space;=&space;\frac{u^T&space;v}{\left&space;\|&space;u&space;\right&space;\|_2&space;\left&space;\|&space;v&space;\right&space;\|_2}=cos(\theta)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?sim(u,&space;v)&space;=&space;\frac{u^T&space;v}{\left&space;\|&space;u&space;\right&space;\|_2&space;\left&space;\|&space;v&space;\right&space;\|_2}=cos(\theta)" title="sim(u, v) = \frac{u^T v}{\left \| u \right \|_2 \left \| v \right \|_2}=cos(\theta)" /></a>

    - <a href="https://www.codecogs.com/eqnedit.php?latex=u^T&space;v" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u^T&space;v" title="u^T v" /></a> is the dot product (or inner product) of two vectors
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\left&space;\|&space;u&space;\right&space;\|_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left&space;\|&space;u&space;\right&space;\|_2" title="\left \| u \right \|_2" /></a> is the norm (or length) of the vector
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" /></a> is the angle between the two vectors

    <img src="Resources/deep_learning/rnn/cosine_sim.png" width=700>

- #### Embedding matrix

  - Notation

    - E = embedding matrix

      <img src="Resources/deep_learning/rnn/embedding_matrix.png" width=300>

    - <a href="https://www.codecogs.com/eqnedit.php?latex=o_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?o_j" title="o_j" /></a> = one-hot vector of word j

    - <a href="https://www.codecogs.com/eqnedit.php?latex=e_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e_j" title="e_j" /></a> = embedding vector of word j

    - <a href="https://www.codecogs.com/eqnedit.php?latex=E\&space;o_j&space;=&space;e_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E\&space;o_j&space;=&space;e_j" title="E\ o_j = e_j" /></a>

  - Goal: learn the embedding matrix

  - Embedding lookup

    In practice, use specialized function to look up an embedding, rather than using matrix-vector multiplication.

### 2.2. Learning word embeddings: Word2vec and GloVe

- #### Learning word embeddings

  - Neural language model

    <img src="Resources/deep_learning/rnn/embedding_neural.png" width=400>

    [Bengio et. al., 2003, A neural probabilistic language model]

  - Other context/target pairs

    <img src="Resources/deep_learning/rnn/embedding_context.png" width=400>

    Context:

    - Last 4 words
    - 4 words on left & right
    - Last 1 word
    - Nearby 1 word

- #### Word2Vec

  - Notation

    - c = context
    - t = targe
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}" title="\hat{y}" /></a> = one-hot vector of softmax output
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\theta_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_t" title="\theta_t" /></a> = parameter associated with output t
    - Vocab size = 10000 for example

  - Skip-grams model

    Come up with a few context-target pairs to create supervised learning problem. Takes as input one context word, tries to predict some word skipping a few words before or after the context word.

    <img src="Resources/deep_learning/rnn/embedding_c2t.png" width=400>

    [Mikolov et. al., 2013. Efficient estimation of word representations in vector space]

    - Softmax: <a href="https://www.codecogs.com/eqnedit.php?latex=p(t|c)=\frac{e^{\theta_t^T&space;e_c&space;}}{\sum_{j=1}^{10000}&space;e^{\theta_j^T&space;e_c}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(t|c)=\frac{e^{\theta_t^T&space;e_c&space;}}{\sum_{j=1}^{10000}&space;e^{\theta_j^T&space;e_c}}" title="p(t|c)=\frac{e^{\theta_t^T e_c }}{\sum_{j=1}^{10000} e^{\theta_j^T e_c}}" /></a>
    - Cost: <a href="https://www.codecogs.com/eqnedit.php?latex=E(\hat{y},&space;y)&space;=&space;\sum_{i=1}^{10000}y_i\&space;log&space;\hat{y}_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(\hat{y},&space;y)&space;=&space;\sum_{i=1}^{10000}y_i\&space;log&space;\hat{y}_i" title="E(\hat{y}, y) = \sum_{i=1}^{10000}y_i\ log \hat{y}_i" /></a>

  - Problems with softmax classification

    (-) Softmax step in the skip-grams model is very expensive to calculate because needing to sum over the entire vocabulary size into the denominator.

    Solutions:

    - Hierarchical softmax classifier (tree)
    - Negative sampling that modifies the training objective to make it run more efficiently

  - Sampling the context c

    There are different heuristics to use in order to balance out the common words (e.g. the, of, and) vs. the less common words.

  - CBow vs. skip-grams

    CBow, the continuous backwards model, takes the surrounding contexts from the middle word, and uses the surrounding words to try to predict the middle word.

- #### Negative sampling

  - Generating training set

    - Pick a context word and a target word, give that a label of 1 or 0. This is one record of the training data.
    - Create a supervised learning problem where the learning algorithm inputs the pair of words (x), and predicts the target label (y) as output.
    - The number of training pairs (k) for each context word is 5 to 20 for smaller data sets, or 2 to 5 for larger data sets.

  - Model

    Turns 10000-way softmax in the skip into 10000 binary classification problem. On every iteration, we're only going to train k + 1 of the binary classifiers (k negative examples and 1 positive examples).

    <img src="Resources/deep_learning/rnn/embedding_c2t_negative.png" width=400>

    [Mikolov et. al., 2013. Distributed representation of words and phrases and their compositionality]

    - Sigmoid: <a href="https://www.codecogs.com/eqnedit.php?latex=p(y=1|c,t)&space;=&space;\sigma(\theta_t^T&space;e_c)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(y=1|c,t)&space;=&space;\sigma(\theta_t^T&space;e_c)" title="p(y=1|c,t) = \sigma(\theta_t^T e_c)" /></a>

  - Negative sampling advantages

    (+) The computation cost is much lower, because updating k + 1 binary classifiers on every iteration is cheaper than updating a 10000-way softmax classifier.

  - Choosing negative examples

    Between sampling according to the empirical frequencies and according to a uniform distribution.

    <a href="https://www.codecogs.com/eqnedit.php?latex=p(w_i)=\frac{f(w_i)^{3/4}}{\sum_{j=1}^{10000}f(w_j)^{3/4}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(w_i)=\frac{f(w_i)^{3/4}}{\sum_{j=1}^{10000}f(w_j)^{3/4}}" title="p(w_i)=\frac{f(w_i)^{3/4}}{\sum_{j=1}^{10000}f(w_j)^{3/4}}" /></a>, where f is empirical frequencies observed in English text

- #### GloVe word vectors

  - GloVe (global vectors for word representation)

    Define context and target as whether or not the two words appear in close proximity (e.g. within ±10 words to each other). <a href="https://www.codecogs.com/eqnedit.php?latex=X_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_{ij}" title="X_{ij}" /></a> is a count that captures how often word i (target t) appears in context of j (context c), i.e. how often they occur close to each other.

    [Pennington et. al., 2014. GloVe: Global vectors for word representation]

  - Model

    Minimize <a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{i=1}^{10000}\sum_{j=1}^{10000}f(X_{ij})(\theta_i^T&space;e_j&plus;b_i&plus;b_j'-logX_{ij})^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{i=1}^{10000}\sum_{j=1}^{10000}f(X_{ij})(\theta_i^T&space;e_j&plus;b_i&plus;b_j'-logX_{ij})^2" title="\sum_{i=1}^{10000}\sum_{j=1}^{10000}f(X_{ij})(\theta_i^T e_j+b_i+b_j'-logX_{ij})^2" /></a>, where

    <a href="https://www.codecogs.com/eqnedit.php?latex=f(X_{ij})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(X_{ij})" title="f(X_{ij})" /></a> = weighting function that assigns

    - <a href="https://www.codecogs.com/eqnedit.php?latex=f(X_{ij})=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(X_{ij})=0" title="f(X_{ij})=0" /></a> if <a href="https://www.codecogs.com/eqnedit.php?latex=X_{ij}=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_{ij}=0" title="X_{ij}=0" /></a>
    - Heuristic choice of weighting that neither gives commonly-occuring words too much weight nor gives the infrequent words too little weight

    <a href="https://www.codecogs.com/eqnedit.php?latex=\theta_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_i" title="\theta_i" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=e_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e_j" title="e_j" /></a> in this particular formulation play symmetric roles

    - Initialize <a href="https://www.codecogs.com/eqnedit.php?latex=\theta_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_i" title="\theta_i" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=e_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e_j" title="e_j" /></a> both uniformly, use gradient descent to minimize the objective, and finally take the average for every word

      <a href="https://www.codecogs.com/eqnedit.php?latex=e_w^{(final)}&space;=&space;\frac{e_w&plus;\theta_w}{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e_w^{(final)}&space;=&space;\frac{e_w&plus;\theta_w}{2}" title="e_w^{(final)} = \frac{e_w+\theta_w}{2}" /></a>

### 2.3. Applications using word embeddings

- #### Sentiment classification

  Challenge: there might not be a huge labeled training set.

  Solution: word embeddings enables to build good sentiment classifiers even with only modest-size label training sets.

  - Sentiment classification problem

    <img src="Resources/deep_learning/rnn/sentiment_ps.png" width=500>

  - Simple sentiment classification model

    <img src="Resources/deep_learning/rnn/sentiment_simple.png" width=560>

    - (+) By using the average operation, this particular algorithm works for reviews that are either short or long.
    - (-) Ignores word order. E.g. by averaging words meaning, incorrectly classifies "Completely lacking in good taste, good service, and good ambience."

  - RNN for sentiment classification

    <img src="Resources/deep_learning/rnn/sentiment_rnn.png" width=560>

    - (+) Takes word sequence into account
    - (+) Word embeddings can be trained from a much larger data set, so the model will do a better job generalizing to new words not seens in the training set

- #### Debiasing word embeddings

  Word embeddings can reflect gender, ethnicity, age, sexual orientation, and other biases of the text used to train the model. Because AI algorithms are increasingly trusted to make extremely important decisions, we want to eliminate undesirable forms of bias.

  [Bolukbasi et. al., 2016. Man is to computer programmer as woman is to homemaker? Debiasing word embeddings]

  1. Identify bias direction.

      Take average of <a href="https://www.codecogs.com/eqnedit.php?latex=e_{he}&space;-&space;e_{she}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e_{he}&space;-&space;e_{she}" title="e_{he} - e_{she}" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=e_{male}&space;-&space;e_{female}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e_{male}&space;-&space;e_{female}" title="e_{male} - e_{female}" /></a>, ..., to calculate the direction of bias

      <img src="Resources/deep_learning/rnn/embedding_bias.png" width=400>

  2. Neutralize: For every word that is not definitional, project to get rid of bias.

      <img src="Resources/deep_learning/rnn/embedding_bias_neutralize.png" width=400>

  3. Equalize pairs: Equalize pairs of words that you might want to have differ only through the e.g. gender property

      <img src="Resources/deep_learning/rnn/embedding_bias_equalize.png" width=800>

## 3. Sequence models & attention mechanism

### 3.1. Various sequence to sequence architectures

- #### Basic models

  - Machine translation

    <img src="Resources/deep_learning/rnn/s2s_translation.png" width=450>

    [Sutskever et al., 2014. Sequence to sequence learning with neural networks] <br>
    [Cho et al., 2014. Learning phrase representations using RNN encoder-decoder for statistical machine translation]

  - Image captioning

    <img src="Resources/deep_learning/rnn/s2s_captioning.png" width=700>

    [Mao et. al., 2014. Deep captioning with multimodal recurrent neural networks] <br>
    [Vinyals et. al., 2014. Show and tell: Neural image caption generator] <br>
    [Karpathy and Li, 2015. Deep visual-semantic alignments for generating image descriptions]

- #### Picking the most likely sentence

  - Machine translation as building a **conditional language model**

    - Language model

      <img src="Resources/deep_learning/rnn/language_model.png" width=270>

      Outputs <a href="https://www.codecogs.com/eqnedit.php?latex=p(y^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;...,&space;y^{\left&space;\langle&space;T_y&space;\right&space;\rangle})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(y^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;...,&space;y^{\left&space;\langle&space;T_y&space;\right&space;\rangle})" title="p(y^{\left \langle 1 \right \rangle}, ..., y^{\left \langle T_y \right \rangle})" /></a>

    - Machine translation

      <img src="Resources/deep_learning/rnn/machine_translation.png" width=400>

      Outputs <a href="https://www.codecogs.com/eqnedit.php?latex=p(y^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;...,&space;y^{\left&space;\langle&space;T_y&space;\right&space;\rangle}&space;|&space;x^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;...,&space;x^{\left&space;\langle&space;T_x&space;\right&space;\rangle})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(y^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;...,&space;y^{\left&space;\langle&space;T_y&space;\right&space;\rangle}&space;|&space;x^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;...,&space;x^{\left&space;\langle&space;T_x&space;\right&space;\rangle})" title="p(y^{\left \langle 1 \right \rangle}, ..., y^{\left \langle T_y \right \rangle} | x^{\left \langle 1 \right \rangle}, ..., x^{\left \langle T_x \right \rangle})" /></a>

  - Finding the most likely translation

    Should not sample output sentence at random. Instead, find the output sentence that maximize the conditional probability:

    <img src="Resources/deep_learning/rnn/s2s_obj.png" width=300>

  - Greedy search vs. approximate search

    Greedy search not a good way to find the most likely translation

    - (-) Not always optimal to compare the conditional probability of one word at a time
    - (-) Possible combination of words is exponentially large

    Approximate search algorithm

    - (-) Not always able to pick the output sentence that maximizes the conditional probability
    - (+) Does a good enough job

### 3.2. Beam search algorithm

- #### Beam search

  Beam search can consider multiple alternatives rather than just one possibility. It runs faster than exact search algorithms, but is not guaranteed to find exact maximum for <a href="https://www.codecogs.com/eqnedit.php?latex=\arg&space;\max_y&space;p(y|x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\arg&space;\max_y&space;p(y|x)" title="\arg \max_y p(y|x)" /></a>.

  - Notation

    - B = beam width, e.g. B = 3 considers 3 most likely possible choices at each step during decoding. When B = 1, beam search becomes greedy search.

  - Step 1

    Find B = 3 most likely possible choices of <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle}" title="\hat{y}^{\left \langle 1 \right \rangle}" /></a> that maximize <a href="https://www.codecogs.com/eqnedit.php?latex=p(\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle}|x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle}|x)" title="p(\hat{y}^{\left \langle 1 \right \rangle}|x)" /></a>

    <img src="Resources/deep_learning/rnn/beam_s1.png" width=400>

  - Step 2

    Find B = 3 most likely possible choices of <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle}" title="\hat{y}^{\left \langle 2 \right \rangle}" /></a> that maximize <a href="https://www.codecogs.com/eqnedit.php?latex=p(\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle}|x,&space;\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle}|x,&space;\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle})" title="p(\hat{y}^{\left \langle 2 \right \rangle}|x, \hat{y}^{\left \langle 1 \right \rangle})" /></a>. Then, find B = 3 most likely possible choices of <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle}" title="\hat{y}^{\left \langle 1 \right \rangle}, \hat{y}^{\left \langle 2 \right \rangle}" /></a> that maximize <a href="https://www.codecogs.com/eqnedit.php?latex=p(\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle}|x)=p(\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle}|x)\&space;p(\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle}|x,&space;\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle}|x)=p(\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle}|x)\&space;p(\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle}|x,&space;\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle})" title="p(\hat{y}^{\left \langle 1 \right \rangle}, \hat{y}^{\left \langle 2 \right \rangle}|x)=p(\hat{y}^{\left \langle 1 \right \rangle}|x)\ p(\hat{y}^{\left \langle 2 \right \rangle}|x, \hat{y}^{\left \langle 1 \right \rangle})" /></a>

    <img src="Resources/deep_learning/rnn/beam_s2.png" width=400>

  - Step 3

    Find B = 3 most likely possible choices of <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}^{\left&space;\langle&space;3&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}^{\left&space;\langle&space;3&space;\right&space;\rangle}" title="\hat{y}^{\left \langle 3 \right \rangle}" /></a> that maximize <a href="https://www.codecogs.com/eqnedit.php?latex=p(\hat{y}^{\left&space;\langle&space;3&space;\right&space;\rangle}|x,&space;\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(\hat{y}^{\left&space;\langle&space;3&space;\right&space;\rangle}|x,&space;\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle})" title="p(\hat{y}^{\left \langle 3 \right \rangle}|x, \hat{y}^{\left \langle 1 \right \rangle}, \hat{y}^{\left \langle 2 \right \rangle})" /></a>. Then, find B = 3 most likely possible choices of <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;3&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;3&space;\right&space;\rangle}" title="\hat{y}^{\left \langle 1 \right \rangle}, \hat{y}^{\left \langle 2 \right \rangle}, \hat{y}^{\left \langle 3 \right \rangle}" /></a> that maximize <a href="https://www.codecogs.com/eqnedit.php?latex=p(\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;3&space;\right&space;\rangle}|x)=p(\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle}|x)\&space;p(\hat{y}^{\left&space;\langle&space;3&space;\right&space;\rangle}|x,&space;\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;3&space;\right&space;\rangle}|x)=p(\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle}|x)\&space;p(\hat{y}^{\left&space;\langle&space;3&space;\right&space;\rangle}|x,&space;\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;\hat{y}^{\left&space;\langle&space;2&space;\right&space;\rangle})" title="p(\hat{y}^{\left \langle 1 \right \rangle}, \hat{y}^{\left \langle 2 \right \rangle}, \hat{y}^{\left \langle 3 \right \rangle}|x)=p(\hat{y}^{\left \langle 1 \right \rangle}, \hat{y}^{\left \langle 2 \right \rangle}|x)\ p(\hat{y}^{\left \langle 3 \right \rangle}|x, \hat{y}^{\left \langle 1 \right \rangle}, \hat{y}^{\left \langle 2 \right \rangle})" /></a>

    <img src="Resources/deep_learning/rnn/beam_s3.png" width=455>

- #### Refinements to beam search

  - Problem with beam search

    <a href="https://www.codecogs.com/eqnedit.php?latex=\arg&space;\max_y&space;\prod_{t=1}^{T_y}p(\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle}|x,&space;\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;...,&space;\hat{y}^{\left&space;\langle&space;t-1&space;\right&space;\rangle})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\arg&space;\max_y&space;\prod_{t=1}^{T_y}p(\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle}|x,&space;\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;...,&space;\hat{y}^{\left&space;\langle&space;t-1&space;\right&space;\rangle})" title="\arg \max_y \prod_{t=1}^{T_y}p(\hat{y}^{\left \langle t \right \rangle}|x, \hat{y}^{\left \langle 1 \right \rangle}, ..., \hat{y}^{\left \langle t-1 \right \rangle})" /></a>

    A product of many small numbers can result in numerical underflow, meaning that it's too small for the floating part representation in the computer to store accurately.

  - Length normalization

    In practice, instead of maximizing the product, maximize the logs of the product:

    <a href="https://www.codecogs.com/eqnedit.php?latex=\arg&space;\max_y\&space;\frac{1}{T_y^{\alpha}}\sum_{t=1}^{T_y}\log&space;p(\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle}|x,&space;\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;...,&space;\hat{y}^{\left&space;\langle&space;t-1&space;\right&space;\rangle})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\arg&space;\max_y\&space;\frac{1}{T_y^{\alpha}}\sum_{t=1}^{T_y}\log&space;p(\hat{y}^{\left&space;\langle&space;t&space;\right&space;\rangle}|x,&space;\hat{y}^{\left&space;\langle&space;1&space;\right&space;\rangle},&space;...,&space;\hat{y}^{\left&space;\langle&space;t-1&space;\right&space;\rangle})" title="\arg \max_y\ \frac{1}{T_y^{\alpha}}\sum_{t=1}^{T_y}\log p(\hat{y}^{\left \langle t \right \rangle}|x, \hat{y}^{\left \langle 1 \right \rangle}, ..., \hat{y}^{\left \langle t-1 \right \rangle})" /></a>

    - (+) More numerically stable algorithm that is less prone to numerical underflow.
    - (+) Normalizing it by the number of words in the translation reduces the penalty for outputting longer translations. <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a> is a hyperparameter that ranges between 0 (no normalization) and 1 (full normalization).

  - Beam width B

    Commonly-used B ranges from 10 to 100.

    - (+) Larger B: more possibilities to consider leads to better results.
    - (-) Larger B: slower and more computationally expensive, because of storing a lot more possibilities around.

- #### Error analysis in beam search

  - Compare <a href="https://www.codecogs.com/eqnedit.php?latex=p(y^*|x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(y^*|x)" title="p(y^*|x)" /></a> vs. <a href="https://www.codecogs.com/eqnedit.php?latex=p(\hat{y}|x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(\hat{y}|x)" title="p(\hat{y}|x)" /></a>, where <a href="https://www.codecogs.com/eqnedit.php?latex=y^*" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y^*" title="y^*" /></a> is the desired result, and <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}" title="\hat{y}" /></a> is the model result that is less desirable.

    - <a href="https://www.codecogs.com/eqnedit.php?latex=p(y^*|x)&space;>&space;p(\hat{y}|x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(y^*|x)&space;>&space;p(\hat{y}|x)" title="p(y^*|x) > p(\hat{y}|x)" /></a>

      Beam search chose <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}" title="\hat{y}" /></a>, but <a href="https://www.codecogs.com/eqnedit.php?latex=y^*" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y^*" title="y^*" /></a> attains higher <a href="https://www.codecogs.com/eqnedit.php?latex=p(y|x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(y|x)" title="p(y|x)" /></a>.

      Conclusion: beam search is at fault.

    - <a href="https://www.codecogs.com/eqnedit.php?latex=p(y^*|x)&space;\leq&space;p(\hat{y}|x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(y^*|x)&space;\leq&space;p(\hat{y}|x)" title="p(y^*|x) \leq p(\hat{y}|x)" /></a>

      <a href="https://www.codecogs.com/eqnedit.php?latex=y^*" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y^*" title="y^*" /></a> is a better translation than <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}" title="\hat{y}" /></a>, but RNN predicted <a href="https://www.codecogs.com/eqnedit.php?latex=p(y^*|x)&space;\leq&space;p(\hat{y}|x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(y^*|x)&space;\leq&space;p(\hat{y}|x)" title="p(y^*|x) \leq p(\hat{y}|x)" /></a>.

      Conclusion: RNN model is at fault.

  - Use error analysis to figure out what fraction of errors are due to beam search vs. RNN model

    <img src="Resources/deep_learning/rnn/beam_error_analysis.png" width=455>

### 3.3. Bleu score

- Problem in evaluating machine translation

  Evaluate a machine translation system if there are multiple equally good answers

- Evaluating machine translation

  Given a machine generated translation, Bleu (Bilingual evaluation understudy) score automatically computes a score that measures how good the machine translation is.

  [Papineni et. al., 2002. Bleu: A method for automatic evaluation of machine translation]

- Notation

  - Clip = give the algorithm credit only up to the maximum number of times that that the n-gram appears in either Reference 1 or Reference 2.
  - <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}" title="\hat{y}" /></a> = machine translation output
  - <a href="https://www.codecogs.com/eqnedit.php?latex=p_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_n" title="p_n" /></a> = Bleu score on n-gram only
  - BP = brevity penalty, penalizes translation systems that output translations that are too short

- Bleu score on n-grams

  <img src="Resources/deep_learning/rnn/bleu.png" width=500>

- Combined Bleu score

  <a href="https://www.codecogs.com/eqnedit.php?latex={BP}\&space;exp(\frac{1}{4}\sum_{i=1}^4&space;p_n)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{BP}\&space;exp(\frac{1}{4}\sum_{i=1}^4&space;p_n)" title="{BP}\ exp(\frac{1}{4}\sum_{i=1}^4 p_n)" /></a>, where
  
  <img src="Resources/deep_learning/rnn/bleu_bp.png" width=500>

- Advantage of Bleu score

  - (+) BLEU score gives a pretty good single real number evaluation metric, so it accelerated the progress of the entire field of machine translation.

### 3.4. Attention model

- #### The problem with long sequences

  In machine translation, difficult for the network to memorize a super long sentence.

  <img src="Resources/deep_learning/rnn/attention_problem.png" width=700>

- #### Attention model

  - t' = time step in the input sentence

  - t = time step in the generated translation

  - <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha^{\left&space;\langle&space;t,t'&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha^{\left&space;\langle&space;t,t'&space;\right&space;\rangle}" title="\alpha^{\left \langle t,t' \right \rangle}" /></a> = attention weight, i.e. amount of attention <a href="https://www.codecogs.com/eqnedit.php?latex=y^{\left&space;\langle&space;t&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y^{\left&space;\langle&space;t&space;\right&space;\rangle}" title="y^{\left \langle t \right \rangle}" /></a> should pay to <a href="https://www.codecogs.com/eqnedit.php?latex=a^{\left&space;\langle&space;t'&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{\left&space;\langle&space;t'&space;\right&space;\rangle}" title="a^{\left \langle t' \right \rangle}" /></a>

    <a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{t'}\alpha^{\left&space;\langle&space;t,&space;t'&space;\right&space;\rangle}&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{t'}\alpha^{\left&space;\langle&space;t,&space;t'&space;\right&space;\rangle}&space;=&space;1" title="\sum_{t'}\alpha^{\left \langle t, t' \right \rangle} = 1" /></a>

  - <a href="https://www.codecogs.com/eqnedit.php?latex=a^{\left&space;\langle&space;t'&space;\right&space;\rangle}&space;=&space;(\overrightarrow{a}^{\left&space;\langle&space;t'&space;\right&space;\rangle},&space;\overleftarrow{a}^{\left&space;\langle&space;t'&space;\right&space;\rangle})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{\left&space;\langle&space;t'&space;\right&space;\rangle}&space;=&space;(\overrightarrow{a}^{\left&space;\langle&space;t'&space;\right&space;\rangle},&space;\overleftarrow{a}^{\left&space;\langle&space;t'&space;\right&space;\rangle})" title="a^{\left \langle t' \right \rangle} = (\overrightarrow{a}^{\left \langle t' \right \rangle}, \overleftarrow{a}^{\left \langle t' \right \rangle})" /></a> concatenated feature vectors from the original sentence

  - c = context that the decoding model pays attention to

    <a href="https://www.codecogs.com/eqnedit.php?latex=c^{\left&space;\langle&space;t&space;\right&space;\rangle}=\sum_{t'}\alpha^{\left&space;\langle&space;t,t'&space;\right&space;\rangle}a^{\left&space;\langle&space;t'&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c^{\left&space;\langle&space;t&space;\right&space;\rangle}=\sum_{t'}\alpha^{\left&space;\langle&space;t,t'&space;\right&space;\rangle}a^{\left&space;\langle&space;t'&space;\right&space;\rangle}" title="c^{\left \langle t \right \rangle}=\sum_{t'}\alpha^{\left \langle t,t' \right \rangle}a^{\left \langle t' \right \rangle}" /></a>

  <img src="Resources/deep_learning/rnn/attention_intuition.png" width=500>

  [Bahdanau et. al., 2014. Neural machine translation by jointly learning to align and translate] <br>
  [Xu et. al., 2015. Show, attend and tell: Neural image caption generation with visual attention]

- #### Computing attention

  <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha^{\left&space;\langle&space;t,t'&space;\right&space;\rangle}=\frac{\exp(e^{\left&space;\langle&space;t,t'&space;\right&space;\rangle})}{\sum_{t'=1}^{T_x}\exp(e^{\left&space;\langle&space;t,t'&space;\right&space;\rangle})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha^{\left&space;\langle&space;t,t'&space;\right&space;\rangle}=\frac{\exp(e^{\left&space;\langle&space;t,t'&space;\right&space;\rangle})}{\sum_{t'=1}^{T_x}\exp(e^{\left&space;\langle&space;t,t'&space;\right&space;\rangle})}" title="\alpha^{\left \langle t,t' \right \rangle}=\frac{\exp(e^{\left \langle t,t' \right \rangle})}{\sum_{t'=1}^{T_x}\exp(e^{\left \langle t,t' \right \rangle})}" /></a>
  
  - <a href="https://www.codecogs.com/eqnedit.php?latex=e^{\left&space;\langle&space;t,t'&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e^{\left&space;\langle&space;t,t'&space;\right&space;\rangle}" title="e^{\left \langle t,t' \right \rangle}" /></a> can be learnt by training a small neural network where the inputs are represented by the hidden state activation in the previous time step <a href="https://www.codecogs.com/eqnedit.php?latex=s^{\left&space;\langle&space;t-1&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s^{\left&space;\langle&space;t-1&space;\right&space;\rangle}" title="s^{\left \langle t-1 \right \rangle}" /></a> and the features from the current time step <a href="https://www.codecogs.com/eqnedit.php?latex=a^{\left&space;\langle&space;t'&space;\right&space;\rangle}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{\left&space;\langle&space;t'&space;\right&space;\rangle}" title="a^{\left \langle t' \right \rangle}" /></a>.

    <img src="Resources/deep_learning/rnn/attention_nn.png" width=200>

    <img src="Resources/deep_learning/rnn/attention_nn2.png" width=350>

  - The algorithm runs in quadratic cost. If we have <a href="https://www.codecogs.com/eqnedit.php?latex=T_x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_x" title="T_x" /></a> words as input and <a href="https://www.codecogs.com/eqnedit.php?latex=T_y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_y" title="T_y" /></a> for the output, the total number of attention parameters is <a href="https://www.codecogs.com/eqnedit.php?latex=T_y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_y" title="T_y" /></a>.

- #### Visualization of attention

  <img src="Resources/deep_learning/rnn/attention_visual.png" width=300>

### 3.5. Speech recognition - audio data

- #### Speech recognition problem

  <img src="Resources/deep_learning/rnn/speech_recognition_problem.png" width=400>

  - Note that phonemes (hand-engineered) representation is no longer necessary, because there are much larger datasets available.
  - Using a spectrogram and optionally a 1D conv layer is a common pre-processing step prior to passing audio data to an RNN, GRU or LSTM.

- #### Attention model for speech recognition

  <img src="Resources/deep_learning/rnn/speech_recognition_attention.png" width=550>

- #### CTC cost for speech recognition

  CTC (Connectionist temporal classification) basic rule: collapse repeated characters not separated by “blank”

  <img src="Resources/deep_learning/rnn/speech_recognition_ctc.png" width=500>

  [Graves et al., 2006. Connectionist Temporal Classification: Labeling unsegmented sequence data with recurrent neural networks]

- #### Trigger word detection

  - Trigger word applications

    <img src="Resources/deep_learning/rnn/trigger_word.png" width=400>

  - Trigger word detection algorithm

    <img src="Resources/deep_learning/rnn/trigger_word_detection.png" width=600>
