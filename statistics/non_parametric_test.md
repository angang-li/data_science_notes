# Non-Parametric Tests

We often use standard hypothesis tests on means of normal distributions to design and analyze experiments. However, it's possible that you will encounter scenarios where you can't rely on only standard tests. This might be due to uncertainty about the true variability of a metric's distribution, a lack of data to assume normality, or wanting to do inference on a statistic that lacks a standard test. It's useful to know about some non-parametric tests, not just as a workaround for cases like this, but also as a second check on your experimental results.

- (+) The main benefit of non-parametric tests is that they don't rely on many assumptions of the underlying population, and so can be used in a wider range of circumstances compared to standard tests.

- [Non-Parametric Tests](#non-parametric-tests)
  - [1. Bootstrapping](#1-bootstrapping)
    - [1.1. Method](#11-method)
    - [1.2. Pros and Cons](#12-pros-and-cons)
    - [1.3. Example](#13-example)
  - [2. Permutation tests](#2-permutation-tests)
    - [2.1. Method](#21-method)
    - [2.2. Example](#22-example)
  - [3. Rank-Sum Test (Mann-Whitney)](#3-rank-sum-test-mann-whitney)
    - [3.1. Method](#31-method)
    - [3.2. Example](#32-example)
  - [4. Sign test](#4-sign-test)
    - [4.1. Method](#41-method)
    - [4.2. Example](#42-example)

## 1. Bootstrapping

### 1.1. Method

- Bootstrapping use resampling of the collected data to make inferences about distributions.
- In a standard bootstrap, a bootstrapped sample means drawing points from the original data with replacement until we get as many points as there were in the original data. Essentially, we're treating the original data as the population: without making assumptions about the original population distribution, using the original data as a model of the population is the best that we can do.
- Taking a lot of bootstrapped samples allows us to estimate the sampling distribution for various statistics on our original data.

### 1.2. Pros and Cons

- (+) The bootstrap procedure is fairly simple and straightforward.
- (+) Since you don't make assumptions about the distribution of data, it can be applicable for any case you encounter.
- (+) The results should also be fairly comparable to standard tests.
- (-) It does take computational effort
- (-) Its output does depend on the data put in. A different sample will produce different intervals, with different accuracies. 
- (-) Confidence intervals coming from the bootstrap procedure will be optimistic compared to the true state of the world. This is because there will be things that we don't know about the real world that we can't account for, due to not having a parametric model of the world's state. Consider the extreme case of trying to understand the distribution of the maximum value: our confidence interval would never be able to include any value greater than the largest observed value and it makes no sense to have any lower bound below the maximum observation. Intuitively, however, there's a pretty clear possibility for there to be unobserved values that are larger than the one we've observed, especially for skewed data.

### 1.3. Example

- Create a 95% confidence interval for the 90th percentile from a dataset of 5000 data points. First of all, we take a bootstrap sample (i.e., draw 5000 points with replacement from the original data), record the 90th percentile, and then repeat this a large number of times, let's say 100 000. From this bunch of bootstrapped 90th percentile estimates, we form our confidence interval by finding the values that capture the central 95% of the estimates (cutting off 2.5% on each tail).

- Code to calculate confidence interval of a quantile

  ```python
  def quantile_ci(data, q, c = .95, n_trials = 1000):
      """
      Compute a confidence interval for a quantile of a dataset using a bootstrap
      method.
      
      Input parameters:
          data: data in form of 1-D array-like (e.g. numpy array or Pandas series)
          q: quantile to be estimated, must be between 0 and 1
          c: confidence interval width
          n_trials: number of bootstrap samples to perform
      
      Output value:
          ci: Tuple indicating lower and upper bounds of bootstrapped
              confidence interval
      """
      
      # initialize storage of bootstrapped sample quantiles
      n_points = data.shape[0]
      sample_qs = []
      
      # For each trial...
      for _ in range(n_trials):
          # draw a random sample from the data with replacement...
          sample = np.random.choice(data, n_points, replace = True)
          
          # compute the desired quantile...
          sample_q = np.percentile(sample, 100 * q)
          
          # and add the value to the list of sampled quantiles
          sample_qs.append(sample_q)
          
      # Compute the confidence interval bounds
      lower_limit = np.percentile(sample_qs, (1 - c)/2 * 100)
      upper_limit = np.percentile(sample_qs, (1 + c)/2 * 100)
      
      return (lower_limit, upper_limit)
  ```

## 2. Permutation tests

### 2.1. Method

- The permutation test is a resampling-type test used to compare the values on an outcome variable between two or more groups.
- In the case of the permutation test, resampling is done on the group labels. The idea here is that, under the null hypothesis, the outcome distribution should be the same for all groups, whether control or experimental. Thus, we can emulate the null by taking all of the data values as a single large group. Applying labels randomly to the data points (while maintaining the original group membership ratios) gives us one simulated outcome from the null.
- The rest is similar to the sampling approach used in a standard hypothesis test, except that we haven't specified a reference distribution to sample from – we're sampling directly from the data we've collected. After applying the labels randomly to all the data and recording the outcome statistic many times, we compare our actual, observed statistic against the simulated statistics. A p-value is obtained by seeing how many simulated statistic values are as or more extreme than the one actually observed, and a conclusion is then drawn.

### 2.2. Example

- Implement a permutation test to test if the 90th percentile of times is statistically significantly smaller for the experimental group, as compared to the control group.

- Code to calculate p-value of a permutation test

  ```python
  def quantile_permtest(x, y, q, alternative = 'less', n_trials = 10_000):
      """
      Compute a p-value from the number of permuted sample differences that are 
      less than or greater than the observed difference, depending on the desired 
      alternative hypothesis.
      
      Input parameters:
          x: 1-D array-like of data for independent / grouping feature as 0s and 1s
          y: 1-D array-like of data for dependent / output feature
          q: quantile to be estimated, must be between 0 and 1
          alternative: type of test to perform, {'less', 'greater'}
          n_trials: number of permutation trials to perform
      
      Output value:
          p: estimated p-value of test
      """
      
      
      # initialize storage of bootstrapped sample quantiles
      sample_diffs = []
      
      # For each trial...
      for _ in range(n_trials):
          # randomly permute the grouping labels
          labels = np.random.permutation(x)
          
          # compute the difference in quantiles
          cond_q = np.percentile(y[labels == 0], 100 * q)
          exp_q  = np.percentile(y[labels == 1], 100 * q)
          
          # and add the value to the list of sampled differences
          sample_diffs.append(exp_q - cond_q)
      
      # compute observed statistic
      cond_q = np.percentile(y[x == 0], 100 * q)
      exp_q  = np.percentile(y[x == 1], 100 * q)
      obs_diff = exp_q - cond_q
      
      # compute a p-value
      if alternative == 'less':
          hits = (sample_diffs <= obs_diff).sum()
      elif alternative == 'greater':
          hits = (sample_diffs >= obs_diff).sum()
      
      return (hits / n_trials)

  quantile_permtest(data['condition'], data['time'], 0.9,
                    alternative = 'less')
  ```

## 3. Rank-Sum Test (Mann-Whitney)

### 3.1. Method

- The rank-sum test, also known as the Mann-Whitney U test, only uses the collected data to test distributions between groups.

- Let's say we draw one value at random from the populations behind each group. The null hypothesis says that there's an equal chance that the larger value is from the first group as the second group; the alternative hypothesis says that there's an unequal chance, which can be specified as one- or two-tailed.

  In order to test this hypothesis, we should look at the data we've collected and see in how many cases values from one group win compared to values in the second. That is, for each data point in the first group, we count how many values in the second group that are smaller than it. (If both values are equal, we count that as a tie, worth +0.5 to the tally.) This number of wins for the first group gives us a value $U$.

  It turns out that $U$ is approximately normally-distributed, given a large enough sample size. If we have $n_1$ data points in the first group and $n_2$ points in the second, then we have a total of $n_1 n_2$ matchups and an equivalent number of victory points to hand out. Under the null hypothesis, we should expect the number of wins to be evenly distributed between groups, and so the expected wins are $\mu_U = \frac{n_1 n_2}{2}$. The variability in the number of wins can be found to be the following equation (assuming no or few ties):

  $$ 
  \sigma_U = \sqrt{\frac{n_1n_2(n_1+n_2+1)}{12}}
  $$

  These $\mu_U$ and $\sigma_U$ values can then be used to compute a standard normal z-score, which generates a p-value.

- There's no resamplng involved; the test is performed only on the data present. The rank-sum test is not a test of any particular statistic, like the mean or median. Instead, it's a test of distributions.

### 3.2. Example

- Code to calculate rank-sum test p-value

  ```python
  def ranked_sum(x, y, alternative = 'two-sided'):
      """
      Return a p-value for a ranked-sum test, assuming no ties.
      
      Input parameters:
          x: 1-D array-like of data for first group
          y: 1-D array-like of data for second group
          alternative: type of test to perform, {'two-sided', less', 'greater'}
      
      Output value:
          p: estimated p-value of test
      """
      
      # compute U
      u = 0
      for i in x:
          wins = (i > y).sum()
          ties = (i == y).sum()
          u += wins + 0.5 * ties
      
      # compute a z-score
      n_1 = x.shape[0]
      n_2 = y.shape[0]
      mean_u = n_1 * n_2 / 2
      sd_u = np.sqrt( n_1 * n_2 * (n_1 + n_2 + 1) / 12 )
      z = (u - mean_u) / sd_u
      
      # compute a p-value
      if alternative == 'two-sided':
          p = 2 * stats.norm.cdf(-np.abs(z))
      if alternative == 'less':
          p = stats.norm.cdf(z)
      elif alternative == 'greater':
          p = stats.norm.cdf(-z)
      
      return p

  ranked_sum(data[data['condition'] == 0]['time'],
             data[data['condition'] == 1]['time'],
             alternative = 'greater')
  ```

- Code to perform rank-sum test using scipy stats package [`mannwhitneyu`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)

  This function considers more factors than the implementation above, including a correction on the standard deviation for ties and a continuity correction (since we're approximating a discretely-valued distribution with a continuous one). In addition, the approach they take is computationally more efficient, based on the sum of value ranks (hence the rank-sum test name) rather than the matchups explanation provided above.

  ```python
  stats.mannwhitneyu(data[data['condition'] == 0]['time'],
                     data[data['condition'] == 1]['time'],
                     alternative = 'greater')
  ```

## 4. Sign test

### 4.1. Method

- The sign test only uses the collected data to compute a test result. It only requires that there be *paired values between two groups to compare*, and tests whether one group's values tend to be higher than the other's.

- In the sign test, we don't care how large differences are between groups, only which group takes a larger value. So comparisons of 0.21 vs. 0.22 and 0.21 vs. 0.31 are both counted equally as a point in favor of the second group. This makes the sign test a fairly weak test, though also a test that can be applied fairly broadly. It's most useful when we have very few observations to draw from and can't make a good assumption of underlying distribution characteristics.

### 4.2. Example

- Can use a sign test as an additional check on click rates that have been aggregated on a daily basis.

  The count of victories for a particular group can be modeled with the binomial distribution. Under the null hypothesis, it is equally likely that either group has a larger value (in the case of a tie, we ignore the comparison): the binomial distribution's success parameter is $p = 0.5$.

- Code to calculate p-value of a sign test

  ```python
  def sign_test(x, y, alternative = 'two-sided'):
      """
      Return a p-value for a ranked-sum test, assuming no ties.
      
      Input parameters:
          x: 1-D array-like of data for first group
          y: 1-D array-like of data for second group
          alternative: type of test to perform, {'two-sided', less', 'greater'}
      
      Output value:
          p: estimated p-value of test
      """
      
      # compute parameters
      n = x.shape[0] - (x == y).sum()
      k = (x > y).sum() - (x == y).sum()

      # compute a p-value
      if alternative == 'two-sided':
          p = min(1, 2 * stats.binom(n, 0.5).cdf(min(k, n-k)))
      if alternative == 'less':
          p = stats.binom(n, 0.5).cdf(k)
      elif alternative == 'greater':
          p = stats.binom(n, 0.5).cdf(n-k)
      
      return p
    
  sign_test(data['control'], data['exp'], 'less')
  ```