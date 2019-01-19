# Spark

- [Spark](#spark)
  - [1. Spark basics](#1-spark-basics)
    - [1.1. Spark vs. Hadoop](#11-spark-vs-hadoop)
    - [1.2. Streaming data](#12-streaming-data)
    - [1.3. Four different modes to setup Spark](#13-four-different-modes-to-setup-spark)
    - [1.4. Spark use cases](#14-spark-use-cases)
    - [1.5. You don't always need Spark](#15-you-dont-always-need-spark)
    - [1.6. Spark's limitations](#16-sparks-limitations)
    - [1.7. Beyond Spark for Storing and Processing Big Data](#17-beyond-spark-for-storing-and-processing-big-data)

## 1. Spark basics

### 1.1. Spark vs. Hadoop

- Hadoop is an older system than Spark but is still used by many companies.

- The major difference between Spark and Hadoop is how they use memory. Hadoop writes intermediate results to disk whereas Spark tries to keep data in memory whenever possible. This makes Spark faster for many use cases. Spark does in memory distributed data analysis, in order to make jobs faster.

- While Spark is great for iterative algorithms, there is not much of a performance boost over Hadoop MapReduce when doing simple counting. Migrating legacy code to Spark, especially on hundreds of nodes that are already in production, might not be worth the cost for the small performance boost.

- Spark does not include a file storage system. You can use Spark on top of HDFS but you do not have to. Spark can read in data from other sources as well such as [Amazon S3](https://aws.amazon.com/s3/).

### 1.2. Streaming data

- The use case is when you want to store and analyze data in real-time such as Facebook posts or Twitter tweets.

- Spark has a streaming library called [Spark Streaming](https://spark.apache.org/docs/latest/streaming-programming-guide.html) although it is not as popular and fast as some other streaming libraries. Other popular streaming libraries include [Storm](http://storm.apache.org/) and [Flink](https://flink.apache.org/).

### 1.3. Four different modes to setup Spark

- Local mode - prototype
- Other three modes - distributed and declares a cluster manager.
    - The **cluster manager** is a separate process that monitors the available resources, and makes sure that all machines are responsive during the job.
    - 3 different options of cluster managers
        - Standalone cluster manager
        - YARN (from Hadoop)
        - Mesos (open source from UC Berkeley's AMPLab Coordinators)

    <img src="resources/spark_modes.png" width=450>

### 1.4. Spark use cases

Here are a few resources about different Spark use cases.

- [Data Analytics](http://spark.apache.org/sql/)
- [Machine Learning](http://spark.apache.org/mllib/)
- [Streaming](http://spark.apache.org/streaming/)
- [Graph Analytics](http://spark.apache.org/graphx/)

### 1.5. You don't always need Spark

- Spark is meant for big data sets that cannot fit on one computer. But you don't need Spark if you are working on smaller data sets.

- Sometimes, you can still use pandas on a single, local machine even if your data set is only a little bit larger than memory. E.g., `pandas` can read data in chunks.

- If the data is already stored in a relational database, you can leverage SQL to extract, filter and aggregate the data. If you would like to leverage `pandas` and SQL simultaneously, you can use libraries such as `SQLAlchemy`, which provides an abstraction layer to manipulate SQL tables with generative Python expressions.

- The most commonly used Python Machine Learning libraries are `scikit-learn` and `TensorFlow` or `PyTorch`.

### 1.6. Spark's limitations

- For streaming data, Spark is slower than native streaming tools such as [Storm](http://storm.apache.org/), [Apex](https://apex.apache.org/), and [Flink](https://flink.apache.org/).

- For machine learning, Spark has limited selection of machine learning algorithms. Currently, Spark only supports algorithms that scale linearly with the input data size. In general, deep learning is not available either, though there are many projects integrate Spark with Tensorflow and other deep learning tools.

### 1.7. Beyond Spark for Storing and Processing Big Data

- Spark is not a data storage system, and there are a number of tools besides Spark that can be used to process and analyze large datasets.

- Sometimes it makes sense to use the power and simplicity of SQL on big data. For these cases, a new class of databases, know as NoSQL and NewSQL, have been developed.

- E.g., newer database storage systems like [HBase](https://hbase.apache.org/) or [Cassandra](http://cassandra.apache.org/); distributed SQL engines like [Impala](https://impala.apache.org/) and [Presto](https://prestodb.io/). Many of these technologies use query syntax.

