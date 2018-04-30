# **Machine Learning**

Takes in features, produces lables.

Machine learning takes in data, and transform the data into **decision surface**, which helps classify future cases.

Train and test on different sets of data, otherwise overfitting. Save ~10% as test data.

<br>

<!-- TOC -->

- [**Machine Learning**](#machine-learning)
- [1. Supervised learning](#1-supervised-learning)
    - [1.1. Naive Bayes](#11-naive-bayes)
        - [* ### Bayes rule](#bayes-rule)
        - [* ### Scikit learn on Gaussian Naive Bayes](#scikit-learn-on-gaussian-naive-bayes)
        - [* ### Accuracy of prediction](#accuracy-of-prediction)
        - [* ### Strengths and weaknesses](#strengths-and-weaknesses)
    - [1.2. Support vector machines](#12-support-vector-machines)
        - [* ### Intro SVM](#intro-svm)
        - [* ### Scikit learn on SVM support vector classifier](#scikit-learn-on-svm-support-vector-classifier)
        - [* ### Kernel trick](#kernel-trick)
        - [* ### Parameters in machine learning](#parameters-in-machine-learning)
        - [* ### Strengths and weaknesses](#strengths-and-weaknesses)

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

* ### Strengths and weaknesses
    * Naive Bayes doesn't account for word order, only looks at word frequency, so phrases with distinct meanings don't work well in Naive Bayes <br>

<br>

## 1.2. Support vector machines
* ### Intro SVM
    SVM learns a linear model. <br>
    The separating line maximizes the distance to nearest points, aka. **margin**, to both classes. This maximizes the **robustness** of prediction. <br>
    SVM maximizes the robustness of prediction on top of correct classification. <br>

* ### Scikit learn on SVM support vector classifier
    ```
    from sklearn.svm import SVC
    clf = SVC(kernel="linear")
    clf.fit(features_train, labels_train)
    labels_predicted = clf.predict(features_test)
    ```

* ### Kernel trick
    The kernel conducts feature transformation, so the new feature space is linearly-separable. In this way, the separation line can be non-linear.

* ### Parameters in machine learning
    These parameters are arguments used to create the classifier before fitting. **Avoid overfitting!**
    * `kernel`
    * `C`, controls tradeoff between smooth decision boundary and classifying training points correctly. <br>
        Large C means more training points correct <br>
        → decision boundary more wiggly
    * `gamma`, defines how far the influence of a single training example reaches. <br>
        High gamma means only points close to the decision bounday have influence <br> 
        → decision boundary more wiggly

* ### Strengths and weaknesses
    * SVM is efficient with complicated domain with clear margin of separation
    * SVM is not efficient in very large dataset because the training time is cubic with data size
    * SVM is not efficient when data have lots of noise
    * Overfitting is possible

