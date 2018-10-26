# Big data and Hadoop

According to IBM: "Every day, 2.5 billion gigabytes of high-velocity data are created in a variety of forms, such as social media posts, information gathered in sensors and medical devices, videos and transaction records"

<!-- TOC -->

- [Big data and Hadoop](#big-data-and-hadoop)
  - [1. What is big data](#1-what-is-big-data)
  - [2. Hadoop](#2-hadoop)
    - [2.1. Core Hadoop](#21-core-hadoop)
    - [2.2. Hadoop ecosystem](#22-hadoop-ecosystem)
    - [2.3. Hadoop distributed file system (HDFS)](#23-hadoop-distributed-file-system-hdfs)
    - [2.4. MapReduce](#24-mapreduce)
  - [3. Running Hadoop](#3-running-hadoop)

<!-- /TOC -->

## 1. What is big data

**Big Data** is a loosely defined term used to describe data sets so large and complex that they become awkward to work with using standard statistical software.

**Big data** is high volume, high velocity, and/or high variety information assets that require new forms of processing to enable enhanced decision making, insight discovery and process optimization.

Challenges with big data

- **Volume** (size of data): Needs a cheaper way to store data reliably and to read and process data efficiently.
- **Velocity** (speed at which data were generated and needs to be processed): Needs to store and process at fast speed.
- **Variety** (difference sources and formats): Unstructured data can be difficult to store and reconcile in traditional systems (SQL)
- **Veracity** (uncertainty)

## 2. Hadoop

### 2.1. Core Hadoop

- Store in Hadoop distributed file system (HDFS)
- Process with MapReduce

  <img src="resources/hadoop.jpg" width=400>

### 2.2. Hadoop ecosystem

Hadoop Ecosystem is a platform or framework which solves big data problems. It is a suite which encompasses a number of services (ingesting, storing, analyzing and maintaining) inside it.

<img src="resources/hadoop_ecosystem.png" width=600>

### 2.3. Hadoop distributed file system (HDFS)

HDFS partitions large datasets and stores files across a network of machines.

- **Daemons** of MapReduce
  - Active NameNode: Holds the metadata for HDFS
  - Standby NameNode: Backup the NameNode
  - DataNode: Stores actual HDFS data blocks

  <br><img src="resources/hadoop_hdfs.png" width=400>

- Resolving node failures

    - Data redundancy: To resolve DataNode failure, Hadoop replicates each block 3 times as it is stored in HDFS
    - NameNode standby: To resolve NameNode failure, configure active NameNode + standby NameNode

- Pros and Cons

  - (+) Handles Terabytes of data
  - (+) Write once - read many times
  - (+) Uses Commodity hardware
  - (-) Not good to low latency access
  - (-) Bad for lots of small files
  - (-) Not for multiple writers

### 2.4. MapReduce

- **Map**: takes a set of data and converts it into another set of data, where individual elements are broken down into tuples (key/value pairs).
- **Reduce**: takes the output from a map as input and combines those data tuples into a smaller set of tuples.

  <img src="resources/hadoop_mapreduce.png" width=400>

- **Daemons** of MapReduce
  - JobTracker: Manages MapReduce jobs, distributes indiviual tasks to machines running the TaskTracker, coordinates MapReduce stages.
  - TaskTracker: Instantiates and monitors individual Map and Reduce tasks.

  <img src="resources/hadoop_mapreduce_daemons.png" width=400>

## 3. Running Hadoop

Running a mapreduce job with the vm alias <br>
`hs {mapper script} {reducer script} {input_file} {output directory}`