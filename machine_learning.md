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
    - [1.4. K-Nearest Neighbors](#14-k-nearest-neighbors)
    - [1.5. AdaBoost ("ensemble method")](#15-adaboost-ensemble-method)
    - [1.6. Random Forest ("ensemble method")](#16-random-forest-ensemble-method)
    - [1.7. Linear Regression](#17-linear-regression)
    - [1.8. Outliers](#18-outliers)
- [2. Unsupervised learning](#2-unsupervised-learning)
    - [2.1. K-Means Clustering](#21-k-means-clustering)
    - [Dimensionality reduction](#dimensionality-reduction)

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
    ```python
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    ```

* ### Accuracy of prediction
    accuracy = number of points classified correctly / all points in test set
    ```python
    accuracy = clf.score(features_test, labels_test)
    ```
    Alternatively
    ```python
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(pred, labels_test)
    ```
* ### Accuracy vs. training set size
    * More training data -> fine-tuned algorithm

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
    ```python
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
    ```python
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

    ```python
    entropy = - Σi (pi) log2(pi) 
    pi: fraction of examples in class i
    ```

    * All examples are same class → entropy = 0
    * Examples are evenly split between classes → entropy = 1

    ```python
    import scipy.stats
    entropy = scipy.stats.entropy([2,1],base=2) # 2 fasts, 1 slow
    ```

* ### Information gain
    ```python
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

## 1.4. K-Nearest Neighbors
* ### Algorithm
    A new point is classified by the popular votes among the k training samples nearest to that query point.

* ### Parameters
    * `n_neighbors` <br>
        k, higher k implies smoother decision surfaces

* ### Scikit learn on k-NN
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
    clf.fit(features_train, labels_train)
    labels_predicted = clf.predict(features_test)
    ```

* ### Strengths and weaknesses
    * k-NN takes no a-prior assumption of data
    * k-NN is simple to interpret
    * k-NN requires high memory because the algorithm stores all training data


## 1.5. AdaBoost ("ensemble method")
**Ensemble methods** are meta-classifiers built from many classifiers, usually decision trees

## 1.6. Random Forest ("ensemble method")
* ### Algorithm
    * The **Forest** is an ensemble of **Decision Trees**, most of the time trained with the **“bagging”** method. 
    * Random forest randomly selects observations and features to builds multiple decision trees, and then merges them together to get a more accurate and stable prediction.

* ### Parameters
    * `n_estimators` <br>
        The number of trees the algorithm builds before taking the maximum voting or taking averages of predictions. In general, a higher number of trees increases the performance and makes the predictions more stable, but it also slows down the computation. (default=10)

    * `max_depth` <br>
        The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    
* ### Scikit learn on RF
    ```python
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(min_samples_split=5)
    clf.fit(features_train, labels_train)
    labels_predicted = clf.predict(features_test)
    ```
* ### Strengths and weaknesses
    * Given enough trees in the forest, the classifier won’t overfit the model.
    * A large number of trees can make the algorithm to slow and ineffective for real-time predictions. Fast to train, but slow to make predictions.

## 1.7. Linear Regression

- ### Scikit learn on Linear Regression
    ```python
    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    reg.fit (x_train, y_train)
    y_pred = reg.predict(x_test)
    reg.coef_ # slope
    reg.intercept_ # intercept
    reg.score(x_train, y_train) # r-squared score
    ```

- ### Performance metrics to evaluate linear regression
  - Minimizing the sum of squared errors

    minimize $\sum error^2$ where $error = actual - predicted$ <br>
    algorithms:

    - Ordinary Least Squares (OLS)
    - Gradient Descent

  - R squared

    How much is the variation in output is explained by variations in input

<br>

- ### Comparison of classification and regression

| output type|discrete (class labels)|continuous (numbers)|
|---|---|---|
|objectives|decision boundary|best fit line|
|evaluation|accuracy|sum of squared error <br> R squared|


## 1.8. Outliers

- ### Causes
  - sensor malfunction -> ignore
  - data entry errors  -> ignore
  - freak event        -> pay attention to

- ### Outlier detection and removal strategy
  - train
  - remove points with largest residual errors (~10% data)
  - re-train


# 2. Unsupervised learning
## 2.1. K-Means Clustering

- ### Theory
  - **Assign** clusters based on cluster centers
  - **Optimize** cluster centers to minimize within-cluster sum of squares (conceptually, minimize the energy of rubber band)
  - Initial placement of centroids is often random but very important in final clusterings

- ### Scikit learn on K-Means Clustering
    [Documentation](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    ```python
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    kmeans.predict(X)
    ```

- ### Parameters

  - `n_clusters`, number of clusters to form as well as the number of centroids to generate
  - `max_iter`, maximum number of iterations of the k-means algorithm for a single run
  - `n_init`, number of time the k-means algorithm will be run with different centroid seeds

- ### Challenges of K-Means
    - Run the algorithm multiple times to account for bad local minimum


## Dimensionality reduction