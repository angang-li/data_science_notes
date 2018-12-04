# **Pandas**

`import pandas as pd` <br>
Pandas takes 2 data structures: DataFrame and Series

<!-- TOC -->

- [**Pandas**](#pandas)
    - [1. Pandas series](#1-pandas-series)
        - [1.1. Creating Series object from list and dictionary](#11-creating-series-object-from-list-and-dictionary)
        - [1.2. Index and values](#12-index-and-values)
        - [1.3. Series addition](#13-series-addition)
    - [2. Pandas DataFrame](#2-pandas-dataframe)
        - [2.1. Creating DataFrame object from Series object, list, dictionary, and NumPy array](#21-creating-dataframe-object-from-series-object-list-dictionary-and-numpy-array)
        - [2.2. Index, values, and cloumns](#22-index-values-and-cloumns)
        - [2.3. Column selection, addition, and deletion](#23-column-selection-addition-and-deletion)
        - [2.4. Row selection, addition, and deletion](#24-row-selection-addition-and-deletion)
    - [3. DataFrame operations](#3-dataframe-operations)
        - [3.1. Math and statistics](#31-math-and-statistics)
        - [3.2. Methods on data cleaning](#32-methods-on-data-cleaning)
        - [3.3. Relational joins](#33-relational-joins)
        - [3.4. Binning](#34-binning)
        - [3.5. Mapping](#35-mapping)
        - [3.6. Iterate over rows](#36-iterate-over-rows)
    - [4. DataFrame I/O](#4-dataframe-io)
        - [4.1. Reading](#41-reading)
        - [4.2. Writing](#42-writing)

<!-- /TOC -->

## 1. Pandas series

### 1.1. Creating Series object from list and dictionary

- #### from list

    ```python
    age_list = [2, 3, 10]
    s = pd.Series(age_list, name = 'pet age')
    ```

    **s:**

    |   |   |
    |---|---|
    |0  |  2|
    |1  |  3|
    |2  | 10|

    Name: pet age, dtype: int64

- #### from list with explicit index

    ```python
    pet_list = ['dog', 'cat', 'parrot']
    color_list = ['black', 'white', 'green']
    s_color = pd.Series(color_list, index = pet_list)
    ```

    **s_color:**

    |   |   |
    |---|---|
    |dog   | black|
    |cat   | white|
    |parrot| green|

    dtype: int64

- #### from dictionary

    ```python
    age_dict = {'dog':2, 'cat':3, 'parrot':10}
    s_age = pd.Series(age_dict)
    ```

    **s_age:**

    |   |   |
    |---|---|
    |dog   |  2|
    |cat   |  3|
    |parrot| 10|

    dtype: int64

### 1.2. Index and values

- #### extract all the index and values
    `s.index` <br>
    `s.values`

- #### select rows by implicit index <br>
    `s[2]` <br>
    `s[2:4]`, inclusive, exclusive <br>
    `s.iloc[2:4]` <br>

- #### select rows by explicit index <br>
    `s['dog']` <br>
    `s['dog':'parrot']`, inclusive, inclusive <br>
    `s.loc['dog':'parrot']` <br>

- #### select rows by boolean mask <br>
    `s[s > 2]` <br>

### 1.3. Series addition

- #### add series s1 and s2

    `s1.add(s2, fill_value=0)`, Fill missing (NaN) values with this value

## 2. Pandas DataFrame

### 2.1. Creating DataFrame object from Series object, list, dictionary, and NumPy array

- #### from a dictionary of Series

    ```python
    df = pd.DataFrame({'age':s_age, 'color':s_color})
    ```

    **df:**

    |    | age | color |
    |--- | --- | ---   |
    |dog |2    | black |
    |cat |3    | white |
    |parrot|10 | green |

- #### from a list

    ```python
    age_list = [2, 3, 10]
    df = pd.DataFrame(age_list)
    ```

    **df:**

    |   |   |
    |---|---|
    |0  |  2|
    |1  |  3|
    |2  | 10|

- #### from a list of list

    ```python
    pet_list = ['dog', 'cat', 'parrot']
    pet_list_list = [[2, 'black'], [3, 'white'], [10, 'green']]
    df = pd.DataFrame(pet_list_list, columns = ['age', 'color'], index = pet_list)
    ```

    **df:**

    |    | age | color |
    |--- | --- | ---   |
    |dog |2    | black |
    |cat |3    | white |
    |parrot|10 | green |

- #### from a list of dictionaries

    ```python
    pet_list = ['dog', 'cat', 'parrot']
    pet_list_dict = [{'age': 2, 'color': 'black'}, 
                     {'age': 3, 'color': 'white'}, 
                     {'age': 10, 'color': 'green'}] # each dict is a different row
    df = pd.DataFrame(pet_list_dict, index = pet_list) # constructor method
    ```

    **df:**

    |    | age | color |
    |--- | --- | ---   |
    |dog |2    | black |
    |cat |3    | white |
    |parrot|10 | green |

- #### from a dictionary of lists

    ```python
    pet_dict = {'type': ['dog', 'cat', 'parrot'], 'age': [2, 3, 10], 
                'color': ['black', 'white', 'green']}
    df2 = pd.DataFrame(pet_dict)
    ```

    **df2:**

    |	|age|	color|	type|
    |---|---|--------|------|
    |0	|2	|black	 |dog   |
    |1	|3	|white	 |cat   |
    |2	|10	|green	 |parrot|

- #### assign explicit index from a column

    ```python
    df = df2.set_index('type')
    ```

    **df:**

    |    | age | color |
    |--- | --- | ---   |
    |dog |2    | black |
    |cat |3    | white |
    |parrot|10 | green |

### 2.2. Index, values, and cloumns

- #### Extract index, values, and columns
    `df.shape` <br>
    `df.index` <br>
    `df.values` <br>
    `df.columns` <br>
    `df.dtypes` <br>

### 2.3. Column selection, addition, and deletion

- #### Select columns
    `df['age']`

- #### Reorganize columns
    `df_reorg = df[['color', 'age']]`

- #### Rename columns
    `df_rename = df.rename(columns = {'age':'Age', 'color':'Color'})`

- #### Create a new column
    `df['age (months)'] = df['age'] * 12`

- #### Delete a column
    `del df['age (months)']` <br>
    alternatively <br>
    `df.pop('age (months)')`

    **df:**

    |    | age | color |
    |--- | --- | ---   |
    |dog |2    | black |
    |cat |3    | white |
    |parrot|10 | green |

### 2.4. Row selection, addition, and deletion

- #### Select rows by implicit index (integer locaction)
    `df.iloc[0:2]`

- #### Select rows by explicit index (label)
    `df.loc['cat':'parrot']`

- #### Add new rows
    `df_to_add = pd.DataFrame({'age': [1, 4]}, index = ['snake', 'turtle'])`
    `df_append = df.append(df_to_add)`

    **df_append:**

    |    | age | color |
    |--- | --- | ---   |
    |dog |2    | black |
    |cat |3    | white |
    |parrot|10 | green |
    |snake |1  | NaN   |
    |turtle|4  | NaN   |

- #### Delete rows by explicit index (label)
    `df_drop = df_append.drop('snake')`

    **df_drop:**

    |    | age | color |
    |--- | --- | ---   |
    |dog |2    | black |
    |cat |3    | white |
    |parrot|10 | green |
    |turtle|4  | NaN   |

## 3. DataFrame operations

### 3.1. Math and statistics

- `df['age'].mean()`
- `df['color'].unique()`
- `df['color'].nunique()`
- `df['age'].sum()`
- `df['age'].sum(axis=1)`, sum across rows
- `df['color'].value_counts()`
- `df.describe()`                   # display a statistical overview of df
- `data_df.describe(include='all')` # include non-numbers
- `df.count()`
- `df.round(2)`
- `df.round({'age': 0})`

### 3.2. Methods on data cleaning

- Drop all rows with missing data <br>
    `df.dropna(how='any')`, 'any' or 'all'<br>

- Fill NaN with 0 <br>
    `df.fillna(0)`

- Convert to numeric <br>
    `pd.to_numeric(df['col1'])` <br>

- Convert multiple columns to numeric
    `df[['col1', 'col2']].apply(pd.to_numeric)`

- Convert columns from onbject to specific data types
    `df.infer_objects()`

- Work with % entries <br>
    `df['percent'].replace('%', '', regex = True).astype('float')`
    
- Replace, clean data <br>
    `df['col1'] = df['col1'].replace({'value1':'value 1', 'val1':'value 1'})` <br>

- Whether there is missing data <br>
    `df.isnull().sum()` <br>

- Group by <br>
    `df.groupby(['col1'])` <br>

- Sort by <br>
    `df.sort_values(by=['col1'], ascending=False)` <br>

- See documentation of built-in methods <br>
    `dir(df)` <br>
    `print(df.first.\_\_doc__)` <br>

- Reset index <br>
    `df.reset_index(drop=True)`, drop=False keeps old index in a new column

### 3.3. Relational joins

- **Normalized** vs **denormalized** tables <br>
    - Normalize: divide data into multiple tables <br>
    - Denormalize: data from multiple tables are combined into one table <br>

    ```python
    df_to_join = pd.DataFrame({'type': ['cat', 'dog', 'parrot', 'mouse'], 
                           'favorite food': ['fish', 'bones', 'sunflower seeds', 'cheese']})
    ```

    **df_to_join:**

    |    | pet  | favorite food   |
    |--- | ---  | ---             |
    |0   |cat   | fish            |
    |1   |dog   | bones           |
    |2   |parrot| sunflower seeds | 
    |3   |mouse | cheese          | 

    **df:**

    |	|age|	color|	type|
    |---|---|--------|------|
    |0	|2	|black	 |dog   |
    |1	|3	|white	 |cat   |
    |2	|10	|green	 |parrot|

    ```python
    pd.merge(df, df_to_join, left_on = 'type', right_on = 'pet', how = 'inner').drop('type', axis = 1)
    ```
    - how: {‘left’, ‘right’, ‘outer’, ‘inner’} <br>
    - left_index = True, right_index = True <br>
    - suffixes = ('_bitcoin', '_dash') <br>

    |	|age|color   |pet   |favorite food  |
    |---|---|--------|------|------         |
    |0	|2	|black	 |dog   |bones          |
    |1	|3	|white	 |cat   |fish           |
    |2	|10	|green	 |parrot|sunflower seeds|

### 3.4. Binning

- Convert continuous variable to categorical variable

    ```python
    bins = np.arange(0, 16, 5)
    labels = ['<5','5-9','10-14']
    s_categories = pd.cut(df['age'], bins = bins, right = False, labels = labels)
    ```

### 3.5. Mapping

- Format a column to currency

    `df['col1'] = df['col1'].map('${:,.2f}'.format)`

- Format a column to number
  
    `df['col1'] = df['col1'].map('{:,}'.format)` <br>
    `df['col1'] = df['col1'].map('{:,.2f}'.format)`

- **Mapping changes the data type!**

### 3.6. Iterate over rows

- Use `iterrows`

    ```python
    for index, row in cities_pd.iterrows():
        target_city = row['City']
        params['address'] = target_city
        print(f"Retrieving Results for Index {index}: {target_city}.")
    ```

## 4. DataFrame I/O

[**Pandas I/O documentation**](https://pandas.pydata.org/pandas-docs/stable/api.html#id12)

### 4.1. Reading

- #### Read csv

    ```python
    df = pd.read_csv(file_path)
    df.head()    # data_df.tail()
    ```

- #### Read sqlite

    ```python
    import sqlite3
    conn = sqlite3.connect('worldbank.db')
    pd.read_sql('SELECT * FROM table1 WHERE year = "2017" AND countrycode = "BRA"', con = conn).head()
    ```

### 4.2. Writing

- #### Export data to Excel
 
    ```python
    writer = pd.ExcelWriter('output.xlsx')
    df.to_excel(writer,'Sheet1')
    writer.save()
    ```

- #### Export data to csv

    ```python
    df.to_csv('output.csv', index=False, header=True)  # index=True, index_label= 'Ind'
    ```

- #### Export data to json

    ```python
    df.to_json('output.json', orient='records')
    ```

- #### Export to sqlite

    ```python
    import sqlite3

    # connect to the database
    # note that sqlite3 will create this database file if it does not exist already
    conn = sqlite3.connect('worldbank.db')
    df.to_sql("table1", con=conn, if_exists='replace', index=False)

    # commit any changes to the database and close the connection to the database
    conn.commit()
    conn.close()
    ```