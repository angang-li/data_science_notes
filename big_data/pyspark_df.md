# Data wrangling with PySpark

- [Data wrangling with PySpark](#data-wrangling-with-pyspark)
  - [1. Creating DataFrame](#1-creating-dataframe)
    - [1.1. Spark session](#11-spark-session)
    - [1.2. Read data file into dataframe](#12-read-data-file-into-dataframe)
    - [1.3. Create dataframe](#13-create-dataframe)
    - [1.4. Write data file from dataframe](#14-write-data-file-from-dataframe)
    - [1.5. Convert between PySpark dataframe and pandas dataframe](#15-convert-between-pyspark-dataframe-and-pandas-dataframe)
  - [2. Exploring DataFrame](#2-exploring-dataframe)
    - [2.1. Overview of dataframe](#21-overview-of-dataframe)
    - [2.2. Access and modify data from dataframe](#22-access-and-modify-data-from-dataframe)
  - [3. DataFrame functions](#3-dataframe-functions)
    - [3.1. Aggregate functions](#31-aggregate-functions)
    - [3.2. User defined functions (UDF)](#32-user-defined-functions-udf)
    - [3.3. Window functions](#33-window-functions)
    - [3.4. Date](#34-date)

## 1. Creating DataFrame

### 1.1. Spark session

The first component of a Spark program is a `SparkContext`, or equivalently `SparkSession` in `pyspark.sql`

- Instantiate Spark session

    ```python
    import pyspark
    from pyspark import SparkConf
    from pyspark.sql import SparkSession

    # Set or update SparkSession parameters
    spark = SparkSession \
    .builder \
    .appName("Our first Python Spark SQL example") \
    .getOrCreate()

    # Check if the change went through
    spark.sparkContext.getConf().getAll()
    ```

- Stop Spark session

    ```python
    # Stop at the end
    spark.stop()
    ```

### 1.2. Read data file into dataframe

- Read json

    ```python
    path = "data/sparkify_log_small.json" # or the data file path of remote cluster
    df = spark.read.json(path)
    ```

- Read csv

    ```python
    # From local
    path = "data/sparkify_log_small.csv"
    df = spark.read.csv(path, header=True)
    ```

    ```python
    # From remote
    from pyspark import SparkFiles
    url = "https://s3.amazonaws.com/zepl-trilogy-test/food.csv"
    spark.sparkContext.addFile(url)
    df = spark.read.csv(SparkFiles.get("food.csv"), sep=",", header=True)
    df.show()
    ```

- Read data with date format

    ```python
    from pyspark import SparkFiles
    url ="https://s3.us-east-2.amazonaws.com/trilogy-dataviz/rainfall.csv"
    spark.sparkContext.addFile(url)
    df = spark.read.csv(SparkFiles.get("rainfall.csv"), sep=",", header=True, inferSchema=True, timestampFormat="yyyy/MM/dd HH:mm:ss")
    df.show()
    ```

- Read data with defined schema

    ```python
    # Import struct fields that we can use
    from pyspark.sql.types import StructField, StringType, IntegerType, StructType

    # Next we need to create the list of struct fields
    schema = [StructField("food", StringType(), True), StructField("price", IntegerType(), True),]

    # Pass in our fields
    final = StructType(fields=schema)

    # Read our data with our new schema
    dataframe = spark.read.csv(SparkFiles.get("food.csv"), sep=",", header=True, schema=final)
    ```

### 1.3. Create dataframe

- Create dataframe from RDD using Row function

    ```python
    # Row functions allow specifying column names for dataframes
    data = sc.parallelize([
        Row(id=1, name="Alice", score=50),
        Row(id=2, name="Bob", score=80)
    ])
    df = data.toDF()
    ```

- Create dataframe from raw data using SQLContext

    ```python
    # SQLContext can create dataframes directly from raw data
    sqlContext = SQLContext(sc)
    data = [
        ('Alice', 50),
        ('Bob', 80),
    ]
    df = sqlContext.createDataFrame(data, ['Name', 'Score'])
    ```

- Create dataframe using SQL Context and the Row function

    ```python
    data = sc.parallelize([
        Row(1, "Alice", 50),
        Row(2, "Bob", 80),
    ])
    column_names = Row('id', 'name', 'score')  
    students = data.map(lambda r: column_names(*r))
    df = sqlContext.createDataFrame(students)
    ```

### 1.4. Write data file from dataframe

- Write csv

    ```python
    out_path = "data/sparkify_log_small.csv"

    # Write as csv
    df.write.save(out_path, format="csv", header=True)
    ```

### 1.5. Convert between PySpark dataframe and pandas dataframe

- Convert pyspark dataframe into pandas dataframe

    ```python
    pandas_df = df.toPandas()
    pandas_df.head()
    ```

- Convert pandas dataframe into pyspark dataframe

    ```python
    sqlContext.createDataFrame(pandas_df).show()
    ```

## 2. Exploring DataFrame

### 2.1. Overview of dataframe

- Print table schema

    ```python
    df.printSchema()
    ```

- Describe data types

    ```python
    df.describe()
    ```

- Show the columns

    ```python
    df.columns
    ```

- Show the first few rows

    ```python
    # Show as a table
    df.show(n=1)
    ```

    ```python
    # Show as a list of rows
    df.take(5)
    ```

    ```python
    # Show the first row
    df.first()
    ```

    ```python
    df.head()
    ```

- Describe summary statistics

    ```python
    # All columns
    df.describe().show()
    ```

    ```python
    # An individual column
    df.describe("artist").show()
    ```

- Check the number of rows

    ```python
    df.count()
    ```

### 2.2. Access and modify data from dataframe

- [Declarative]: create a view to run SQL queries

    ```python
    df.createOrReplaceTempView("df_table")
    ```

- Select column(s)

    ```python
    # Select 1 column
    df['price'] # is of type pyspark.sql.column.Column
    df.select('price') # is of type pyspark.sql.dataframe.DataFrame
    df.select('price').show() # show selected data
    ```

    ```python
    # Select multiple columns
    df.select(["age", "height_meter", "weight_kg"]).show()
    ```

    ```python
    # Collect a column as a list
    df.select("price").collect()
    ```

    ```python
    # Select with "where" condition
    df.select(['userId', 'firstName']).where(df.userID == "1046").collect()
    ```

- [Declarative]: select clause with SQL syntax

    ```python
    spark.sql(
        '''
        SELECT *
        FROM df_table
        WHERE userID == '1046'
        LIMIT 2
        '''
    ).show() # or .collect()
    ```

- Filter rows with given condition

    ```python
    # Filter using SQL syntax
    df.filter("price<20").show()
    ```

    ```python
    # Filter using Python syntax
    df.filter(df["price"] < 200).show()
    df.filter( (df["price"] < 200) | (df['points'] > 80) ).show()
    df.filter(df["country"] == "US").show()
    df.filter(df['userId'] != "")
    ```

- Drop NaN

    ```python
    df_valid = df.dropna(how='any', subset=['userId', 'sessionId'])
    df_valid.count()
    ```

- Drop duplicates

    ```python
    df.select("page").dropDuplicates().sort("page").show()
    ```

- Edit column

    ```python
    # Edit using map function
    complex_data_df.rdd\
        .map(lambda x: (x.col_string + " Boo"))\
        .collect()
    ```

- Add new column

    ```python
    # Add a new column
    df = df.withColumn('newprice', df['price'])
    ```

    ```python
    # Add a new column with calculation
    df = df.withColumn('doubleprice',df['price']*2)
    ```

    ```python
    # Add a new column with built-in function
    df = df.withColumn("Desc", concat(col("Title"), lit(' '), col("Body")))
    ```

- Update column name

    ```python
    # Rename a column
    df.withColumnRenamed('price','newerprice').show()
    ```

    ```python
    # Equivalently, set alias
    df.select(df.price.alias("newerprice")).show()
    ```

- Group by

    ```python
    # Find the average precipitation per year
    averages = df.groupBy("year").avg()
    averages.orderBy("year").select("year", "avg(prcp)").show()
    ```

- Order by

    ```python
    # Order a dataframe by ascending values
    df.orderBy(df["points"].asc()).head(5)
    ```

    equivalently

    ```python
    # Order a dataframe by ascending values
    from pyspark.sql.functions import asc
    df.orderBy(asc("points")).head(5)
    ```

    equivalently

    ```python
    # Order a dataframe by ascending values
    df.sort("points").head(5)
    ```

- Filter, group by, order by

    ```python
    songs_in_hour = df.filter(df.page == "NextSong").groupby(df.hour).count().orderBy(df.hour.cast("float"))
    songs_in_hour.show()
    ```

## 3. DataFrame functions

### 3.1. Aggregate functions

Spark SQL provides built-in methods for the most common aggregations such as `count()`, `countDistinct()`, `avg()`, `max()`, `min()`, etc. in the pyspark.sql.functions module. These methods are not the same as the built-in methods in the Python Standard Library

- Take average

    ```python
    # Use avg
    from pyspark.sql.functions import avg
    df.select(avg("points")).show()
    ```

    equivalently

    ```python
    # Use agg
    df.agg({"points": "avg"}).show()
    ```

    equivalently

    ```python
    # Use agg with avg
    df.agg(avg("points")).show()
    ```

### 3.2. User defined functions (UDF)

The default type of the returned variable for UDFs is string. If we would like to return an other type we need to explicitly do so by using the different types from the pyspark.sql.types module.

- Output string

    ```python
    # Add a new column based on user-defined function
    from pyspark.sql.functions import udf
    import datetime

    get_hour = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0). hour)
    df = df.withColumn("hour", get_hour(df.ts))
    ```

- Output integer

    ```python
    # Add a new column based on user-defined function
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType
    from pyspark.sql.types import IntegerType

    flag_downgrade_event = udf(lambda x: 1 if x == "Submit Downgrade" else 0, IntegerType())
    df = df.withColumn("downgraded", flag_downgrade_event("page"))
    ```

- [Declarative]: UDF with SQL syntax

    ```python
    # User-defined function
    spark.udf.register("get_hour", lambda x: int(datetime.datetime.fromtimestamp(x / 1000.0).hour))

    # SQL query using the user-defined function
    songs_in_hour = spark.sql(
        '''
        SELECT get_hour(ts) AS hour, COUNT(*) as plays_per_hour
        FROM df_table
        WHERE page = "NextSong"
        GROUP BY hour
        ORDER BY cast(hour as int) ASC
        '''
    )
    ```

### 3.3. Window functions

Window functions are a way of combining the values of ranges of rows in a dataframe. When defining the window we can choose how to sort and group (with the partitionBy method) the rows and how wide of a window we'd like to use (described by rangeBetween or rowsBetween).

For further information see the [Spark SQL, DataFrames and Datasets Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html) and the [Spark Python API Docs](https://spark.apache.org/docs/latest/api/python/index.html).

- Cumulative sum

    ```python
    from pyspark.sql import Window
    from pyspark.sql.functions import desc
    from pyspark.sql.functions import sum as Fsum

    # Create window function
    windowval = Window.partitionBy("userId").orderBy(desc("ts")).rangeBetween(Window.unboundedPreceding, 0)

    # Add a column of cumulative sum
    df = df.withColumn("phase", Fsum("downgraded").over(windowval))
    ```

- [Declarative]: cumulative sum with SQL syntax

    ```python
    spark.sql(
        '''
        SELECT userId, ts, home_flag
        SUM(home_flag) OVER(PARTITION BY userId ORDER BY ts DESC RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) phase
        FROM df_table
        '''
    )
    ```

### 3.4. Date

- Show the year and month for the date column

    ```python
    # Import date time functions
    from pyspark.sql.functions import year, month

    # Show the year for the date column
    df.select(year(df["date"])).show()

    # Show the month
    df.select(month(df['Date'])).show()
    ```
