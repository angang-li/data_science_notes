# **Machine Learning**

### Takes in features, produces lables

### Machine learning takes in data, and transform the data into decision surface, which helps classify future cases.

### Train and test on different sets of data, otherwise overfitting. Save ~10% as test data

<br>

<!-- TOC -->

- [**Machine Learning**](#machine-learning)
- [1. Supervised learning](#1-supervised-learning)
    - [1.1. Naive Bayes](#11-naive-bayes)
    - [1.2. Support vector machines](#12-support-vector-machines)

<!-- /TOC -->

<br>

# 1. Supervised learning

Train with examples that have correct answers

## 1.1. Naive Bayes

* ### Bayes rule
    sensitivity: true positive <br>
    specitivity: true negative <br>
    <br>
    
    joint probability = prior probability * evidence <br>
    posterior probability = joint probability / normalizer <br>
    <br>

* ### Scikit learn on Gaussian Naive Bayes
    ```
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    ```

* ### Accuracy of prediction
    accuracy = number of points classified correctly / all points in test set
    ```
    accuracy = clf.score(features_test, labels_test)
    ```
    Alternatively
    ```
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(pred, labels_test)
    ```

* ### Strengths and weakness
    * Naive Bayes doesn't account for word order, only looks at word frequency, so phrases with distinct meanings don't work well in Naive Bayes <br>

<br>

## 1.2. Support vector machines

