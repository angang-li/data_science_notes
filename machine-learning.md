# **Machine Learning**

Takes in features, produces lables.

Machine learning takes in data, and transform the data into **decision surface**, which helps classify future cases.

Train and test on different sets of data, otherwise overfitting. Save ~10% as test data.

<br>

<!-- TOC -->

- [**Machine Learning**](#machine-learning)
- [1. Supervised learning](#1-supervised-learning)
    - [1.1. Naive Bayes](#11-naive-bayes)
    - [1.2. Support Vector Machines](#12-support-vector-machines)
    - [1.3. Decision Trees](#13-decision-trees)

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

## 1.2. Support Vector Machines
* ### Intro SVM
    * SVM learns a linear model.
    * The separating line maximizes the distance to nearest points, aka. **margin**, to both classes. This maximizes the **robustness** of prediction.
    * SVM maximizes the robustness of prediction on top of correct classification.
    * SVM uses kernel tricks to change from linear to non-linear decision surfaces.

* ### Scikit learn on SVM support vector classifier
    ```
    from sklearn.svm import SVC
    clf = SVC(kernel="linear")
    clf.fit(features_train, labels_train)
    labels_predicted = clf.predict(features_test)
    ```

* ### Kernel trick
    The kernel conducts feature transformation, so the new feature space is linearly-separable. In this way, the separation line can be non-linear.

* ### Parameters
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

## 1.3. Decision Trees
* ### Intro Decision Trees
    * Decision trees use the tree structure to do non-linear decision making with simple linear decision surfaces.
    * Decision trees ask multiple linear questions one after another.

* ### Scikit learn on Decision Trees
    ```
    from sklearn import tree
    clf = tree.DecisionTreeClassifier() # default criterion is gini instead of entropy
    clf = clf.fit(features_train, labels_train)
    labels_predicted = clf.predict(features_test)
    ```

* ### Parameters
    * `min_samples_split`, the minimum number of samples required to split an internal node (default=2)

* ### Data impurity and entropy
    * Entropy controls how a decision tree decides where to split the data
    * Entropy is a measure of impurity in a bunch of examples

    * Decision trees make decisions by finding variables and split points along the variables that can make the subset as pure as possible

    ```
    entropy = - Σi (pi) log2(pi) 
    pi: fraction of examples in class i
    ```

    * All examples are same class → entropy = 0
    * Examples are evenly split between classes → entropy = 1

    ```
    import scipy.stats
    entropy = scipy.stats.entropy([2,1],base=2) # 2 fasts, 1 slow
    ```

* ### Information gain
    ```
    information gain = entropy(parent) - weighted average of entropy(children)
    ```
    * Decision tree algorithm maximizes information gain

* ### Bias-Variance dilemma
    * high bias: ignores data, does not learn anything
    * high variance: extremely perceptive to data, only able to replicate what it's seen before

* ### Strengths and weaknesses
    * DT is graphically easy to understand
    * DT is prone to overfitting, especially with lots of features
    * DT is able to build bigger classifiers out of DT in **ensemble methods**

