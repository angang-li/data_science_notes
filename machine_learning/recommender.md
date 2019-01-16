# Recommendation Engines

Three main braches of recommenders

- Knowledge Based Recommendations
- Collaborative Filtering Based Recommendations
  - Model Based Collaborative Filtering
  - Neighborhood Based Collaborative Filtering
- Content Based Recommendations

Often blended techniques of all three types are used in practice to provide the the best recommendation for a particular circumstance.

## 1. Knowledge based recommendation

**Knowledge based recommendation:** recommendation system in which knowledge about the item or user preferences are used to make a recommendation. Knowledge based recommendations frequently are implemented using filters, and are extremely common amongst luxury based goods. Often a rank based algorithm is provided along with knowledge based recommendations to bring the most popular items in particular categories to the user's attention.

**Rank based recommendation:** recommendation system based on higest ratings, most purchases, most listened to, etc.

## 2. Collaborative filtering based recommendation - Neighborhood based

**Collaborative filtering:** a method of making recommendations based on the interactions between users and items.
  
**Neighborhood based collaborative filtering** is used to identify items or users that are "neighbors" with one another.

It is worth noting that two vectors could be similar by similarity metrics while being incredibly, incredibly different by distance metrics. Understanding your specific situation will assist in understanding whether your metric is appropriate.

### 2.1. Similarity based methods

<img src="Resources/recommender/cf_nb_sim.png" width=500>

- **Pearson's correlation coefficient**

    Pearson's correlation coefficient is a measure related to the strength and direction of a *linear* relationship. If we have two vectors x and y, we can compare their individual elements in the following way to calculate Pearson's correlation coefficient:

    <a href="https://www.codecogs.com/eqnedit.php?latex=$$CORR(\textbf{x},&space;\textbf{y})&space;=&space;\frac{\sum\limits_{i=1}^{n}(x_i&space;-&space;\bar{x})(y_i&space;-&space;\bar{y})}{\sqrt{\sum\limits_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum\limits_{i=1}^{n}(y_i-\bar{y})^2}}&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$CORR(\textbf{x},&space;\textbf{y})&space;=&space;\frac{\sum\limits_{i=1}^{n}(x_i&space;-&space;\bar{x})(y_i&space;-&space;\bar{y})}{\sqrt{\sum\limits_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum\limits_{i=1}^{n}(y_i-\bar{y})^2}}&space;$$" title="$$CORR(\textbf{x}, \textbf{y}) = \frac{\sum\limits_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum\limits_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum\limits_{i=1}^{n}(y_i-\bar{y})^2}} $$" /></a>

    where

    <a href="https://www.codecogs.com/eqnedit.php?latex=$$\bar{x}&space;=&space;\frac{1}{n}\sum\limits_{i=1}^{n}x_i$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\bar{x}&space;=&space;\frac{1}{n}\sum\limits_{i=1}^{n}x_i$$" title="$$\bar{x} = \frac{1}{n}\sum\limits_{i=1}^{n}x_i$$" /></a>

- **Spearman's correlation coefficient**

    Spearman's correlation is what is known as a [non-parametric](https://en.wikipedia.org/wiki/Nonparametric_statistics) statistic, which is a statistic whose distribution doesn't depend on parameters. (Statistics that follow normal distributions or binomial distributions are examples of parametric statistics.) Frequently non-parametric statistics are based on the ranks of data rather than the original values collected. If we map each of our data to ranked data values:

    <a href="https://www.codecogs.com/eqnedit.php?latex=$$\textbf{x}&space;\rightarrow&space;\textbf{x}^{r}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\textbf{x}&space;\rightarrow&space;\textbf{x}^{r}$$" title="$$\textbf{x} \rightarrow \textbf{x}^{r}$$" /></a> <br>
    <a href="https://www.codecogs.com/eqnedit.php?latex=$$\textbf{y}&space;\rightarrow&space;\textbf{y}^{r}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\textbf{y}&space;\rightarrow&space;\textbf{y}^{r}$$" title="$$\textbf{y} \rightarrow \textbf{y}^{r}$$" /></a>

    Here, we let the **r** indicate these are ranked values (this is not raising any value to the power of r). Then we compute Spearman's correlation coefficient as:

    <a href="https://www.codecogs.com/eqnedit.php?latex=$$SCORR(\textbf{x},&space;\textbf{y})&space;=&space;\frac{\sum\limits_{i=1}^{n}(x^{r}_i&space;-&space;\bar{x}^{r})(y^{r}_i&space;-&space;\bar{y}^{r})}{\sqrt{\sum\limits_{i=1}^{n}(x^{r}_i-\bar{x}^{r})^2}\sqrt{\sum\limits_{i=1}^{n}(y^{r}_i-\bar{y}^{r})^2}}&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$SCORR(\textbf{x},&space;\textbf{y})&space;=&space;\frac{\sum\limits_{i=1}^{n}(x^{r}_i&space;-&space;\bar{x}^{r})(y^{r}_i&space;-&space;\bar{y}^{r})}{\sqrt{\sum\limits_{i=1}^{n}(x^{r}_i-\bar{x}^{r})^2}\sqrt{\sum\limits_{i=1}^{n}(y^{r}_i-\bar{y}^{r})^2}}&space;$$" title="$$SCORR(\textbf{x}, \textbf{y}) = \frac{\sum\limits_{i=1}^{n}(x^{r}_i - \bar{x}^{r})(y^{r}_i - \bar{y}^{r})}{\sqrt{\sum\limits_{i=1}^{n}(x^{r}_i-\bar{x}^{r})^2}\sqrt{\sum\limits_{i=1}^{n}(y^{r}_i-\bar{y}^{r})^2}} $$" /></a>

    where

    <a href="https://www.codecogs.com/eqnedit.php?latex=$$\bar{x}^r&space;=&space;\frac{1}{n}\sum\limits_{i=1}^{n}x^r_i$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\bar{x}^r&space;=&space;\frac{1}{n}\sum\limits_{i=1}^{n}x^r_i$$" title="$$\bar{x}^r = \frac{1}{n}\sum\limits_{i=1}^{n}x^r_i$$" /></a>

- **Kendall's Tau**

    Kendall's tau is quite similar to Spearman's correlation coefficient. Both of these measures are non-parametric measures of a relationship. Specifically both Spearman and Kendall's coefficients are calculated based on ranking data and not the raw data.

    Similar to both of the previous measures, Kendall's Tau is always between -1 and 1, where -1 suggests a strong, negative relationship between two variables and 1 suggests a strong, positive relationship between two variables.

    Though Spearman's and Kendall's measures are very similar, there are statistical advantages to choosing Kendall's measure in that Kendall's Tau has smaller variability when using larger sample sizes.  However Spearman's measure is more computationally efficient, as Kendall's Tau is O(n^2) and Spearman's correlation is O(nLog(n)). You can find more on this topic in [this thread](https://www.researchgate.net/post/Does_Spearmans_rho_have_any_advantage_over_Kendalls_tau).

    We want to map our data to ranks:

    <a href="https://www.codecogs.com/eqnedit.php?latex=$$\textbf{x}&space;\rightarrow&space;\textbf{x}^{r}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\textbf{x}&space;\rightarrow&space;\textbf{x}^{r}$$" title="$$\textbf{x} \rightarrow \textbf{x}^{r}$$" /></a> <br>
    <a href="https://www.codecogs.com/eqnedit.php?latex=$$\textbf{y}&space;\rightarrow&space;\textbf{y}^{r}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\textbf{y}&space;\rightarrow&space;\textbf{y}^{r}$$" title="$$\textbf{y} \rightarrow \textbf{y}^{r}$$" /></a>

    Then we calculate Kendall's Tau as:

    <a href="https://www.codecogs.com/eqnedit.php?latex=$$TAU(\textbf{x},&space;\textbf{y})&space;=&space;\frac{2}{n(n&space;-1)}\sum_{i&space;<&space;j}sgn(x^r_i&space;-&space;x^r_j)sgn(y^r_i&space;-&space;y^r_j)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$TAU(\textbf{x},&space;\textbf{y})&space;=&space;\frac{2}{n(n&space;-1)}\sum_{i&space;<&space;j}sgn(x^r_i&space;-&space;x^r_j)sgn(y^r_i&space;-&space;y^r_j)$$" title="$$TAU(\textbf{x}, \textbf{y}) = \frac{2}{n(n -1)}\sum_{i < j}sgn(x^r_i - x^r_j)sgn(y^r_i - y^r_j)$$" /></a>

    Where $sgn$ takes the the sign associated with the difference in the ranked values.  An alternative way to write 

    <a href="https://www.codecogs.com/eqnedit.php?latex=$$sgn(x^r_i&space;-&space;x^r_j)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$sgn(x^r_i&space;-&space;x^r_j)$$" title="$$sgn(x^r_i - x^r_j)$$" /></a>

    is in the following way:

    <a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;\begin{cases}&space;-1&space;&&space;x^r_i&space;<&space;x^r_j&space;\\&space;0&space;&&space;x^r_i&space;=&space;x^r_j&space;\\&space;1&space;&&space;x^r_i&space;>&space;x^r_j&space;\end{cases}&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;\begin{cases}&space;-1&space;&&space;x^r_i&space;<&space;x^r_j&space;\\&space;0&space;&&space;x^r_i&space;=&space;x^r_j&space;\\&space;1&space;&&space;x^r_i&space;>&space;x^r_j&space;\end{cases}&space;$$" title="$$ \begin{cases} -1 & x^r_i < x^r_j \\ 0 & x^r_i = x^r_j \\ 1 & x^r_i > x^r_j \end{cases} $$" /></a>

    Therefore the possible results of 

    <a href="https://www.codecogs.com/eqnedit.php?latex=$$sgn(x^r_i&space;-&space;x^r_j)sgn(y^r_i&space;-&space;y^r_j)$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$sgn(x^r_i&space;-&space;x^r_j)sgn(y^r_i&space;-&space;y^r_j)$$" title="$$sgn(x^r_i - x^r_j)sgn(y^r_i - y^r_j)$$" /></a>

    are only 1, -1, or 0, which are summed to give an idea of the proportion of times the ranks of **x** and **y** are pointed in the right direction.

### 2.2. Distance based methods

<img src="Resources/recommender/cf_nb_dis.png" width=500> <br>
[This is a great article](http://dataaspirant.com/2015/04/11/five-most-popular-similarity-measures-implementation-in-python/) on some popular distance metrics.

Note: It is important to have all data be in the same scale. E.g., if some measures are on a 5 point scale, while others are on a 100 point scale.

- **Euclidean Distance**

    Euclidean distance can be considered as straight-line distance between two vectors. For two vectors **x** and **y**, we can compute this as:

    <a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;EUC(\textbf{x},&space;\textbf{y})&space;=&space;\sqrt{\sum\limits_{i=1}^{n}(x_i&space;-&space;y_i)^2}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;EUC(\textbf{x},&space;\textbf{y})&space;=&space;\sqrt{\sum\limits_{i=1}^{n}(x_i&space;-&space;y_i)^2}$$" title="$$ EUC(\textbf{x}, \textbf{y}) = \sqrt{\sum\limits_{i=1}^{n}(x_i - y_i)^2}$$" /></a>

- Manhattan Distance

    Different from euclidean distance, Manhattan distance is a 'manhattan block' distance from one vector to another.  Therefore, you can imagine this distance as a way to compute the distance between two points when you are not able to go through buildings.

    Specifically, this distance is computed as:

    <a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;MANHATTAN(\textbf{x},&space;\textbf{y})&space;=&space;\sqrt{\sum\limits_{i=1}^{n}|x_i&space;-&space;y_i|}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;MANHATTAN(\textbf{x},&space;\textbf{y})&space;=&space;\sqrt{\sum\limits_{i=1}^{n}|x_i&space;-&space;y_i|}$$" title="$$ MANHATTAN(\textbf{x}, \textbf{y}) = \sqrt{\sum\limits_{i=1}^{n}|x_i - y_i|}$$" /></a>

    <img src="Resources/recommender/cf_nb_distances.png" width=200> <br>
    The blue line gives the Manhattan distance, while the green line gives the Euclidean distance between two points.

### 2.3. Making recommendations

- User-based collaborative filtering

    In this type of recommendation, users related to the user you would like to make recommendations for are used to create a recommendation.

    - A simple method

        1. Find movies of neighbors, remove movies that the user has already seen.
        2. Find movies whose ratings are high.
        3. Recommend movies to each user where both 1 and 2 above hold.

    - Other methods for making recommendations using collaborative filtering are based on weighting of the neighbors' ratings based on the 'closeness' of the neighbors.

    - Learn more

        [Domino Data Lab Paper](https://blog.dominodatalab.com/recommender-systems-collaborative-filtering/) <br>
        [Semantic Scholar Paper On Weighted Ratings](https://pdfs.semanticscholar.org/3e9e/bcd9503ef7375c7bb334511804d1e45127e9.pdf)

- Item-based collaborative filtering

    In this type of recommendation, first find the items that are most related to each other item (based on similar ratings). Then you can use the ratings of an individual on those similar items to understand if a user will like the new item.
