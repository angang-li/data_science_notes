# Statistics

## 1. Descriptive statistics (summarize)
* ### 1.1. Measures of central tendency
    * mean       
    * median         
    * mode     

    | | pros | cons |
    |---|---|---|
    |mean| use all numbers, easy |susceptible to outliers|
    |median| robust to outliers | only reflects 1-2 values of the dataset|
    |mode|good for nominal or categorical data, or multiple clusters|less meaningful in small dataset| <br>

    `from stats import mean, median, mode, multi_mode`

* ### 1.2. Measures of variations
    * variance
    * standard deviation
    * z-score:   how far one value is from the mean

        These statistics are susceptible to outliers <br>
        Okay to drop outliers when ... <br>
        Not okay to drop outliers when ... <br>

    * interquartile range (IQR): 3rd quartile - 1st quartile

        IQR as automatic way to identify outliers <br>
        lower outliers: Q1 - 1.5 * IQR <br>
        upper outliers: Q3 + 1.5 * IQR <br>

        `plt.boxplot(arr, showmeans=True)` <br>
        median, IQR, whisker shows data excluding outliers, outliers plotted as individual points

## 2. Inferential statistics (generalize)
* ### 2.1. Sample vs. population
    * sample skewness + fundamental randomness
    * **sampling distribution** is the dist of a stat across an infinite number of samples

* ### 2.2. Standard error of the mean - a measure of confidence
    * SE = s/sqrt(n)
    * s: standard deviation of sample
    * n: sample size

        `from scipy.stats import sem` <br>
        `plt.errorbar(x_axis, means, yerr = standard_errors, fmt='o')` <br>

        comparing error bars could infer whether samples were drawn from the same population

* ### 2.3. The Student's t-test
    how likely that two samples represent the same underlying population

    `from scipy.stats import sem, ttest_ind`
    `(t_stat, p) = ttest_ind(high_prices, low_prices, equal_var=False)`

* ### 2.4. One Sample T-Test
    compare sample mean to the population mean

    Assumptions
    * Data is normally distributed
    * Data is independent
    * Data is randomly sampled

    ```python
    import scipy.stats as stats
    stats.ttest_1samp(sample, population.mean())
    ```

* ### 2.5. Independent T-Test
    compare the means of 2 independent populations

    Assumptions
    * Data is normally distributed
    * Data is independent
    * Data is homogenous (The standard deviations are roughly equal)

    ```python
    import scipy.stats as stats
    # Note: Setting equal_var=False performs Welch's t-test which does 
    # not assume equal population variance
    stats.ttest_ind(population1, population2, equal_var=False)
    ```

* ### 2.6. ANOVA
    are any of the treatments significantly different than the rest?

    ```python
    import scipy.stats as stats
    # Perform the ANOVA
    stats.f_oneway(group1, group2, group3, group4, group5)
    ```

* ### 2.7. Chi-Square
    whether there is a significant difference between the expected frequencies and the observed frequencies in one or more categories

    ```python
    import scipy.stats as stats

    # The degree of freedom is assumed 2
    # With a p-value of 0.05, the confidence level is 1.00-0.05 = 0.95.
    critical_value = stats.chi2.ppf(q = 0.95, df = 2)

    # Run the chi square test with stats.chisquare()
    stats.chisquare(df['observed'], df['expected'])
    ```

## 3. Predictive statistics (unknown)
* ### 3.1. Regression
    `from scipy.stats import linregress`
    `(slope, intercept, _, _, _) = linregress(x_data, y_data)`
    `y_pred = slope * x_data + intercept`

