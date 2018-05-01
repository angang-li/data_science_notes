# **Pandas**
`import pandas as pd` <br>
Pandas takes 2 data structures: DataFrame and Series

<br>

<!-- TOC -->

- [**Pandas**](#pandas)
- [1. Pandas series](#1-pandas-series)
    - [1.1. Creating Series object from list and dictionary](#11-creating-series-object-from-list-and-dictionary)
    - [1.2. Index and values](#12-index-and-values)
- [2. Pandas DataFrame](#2-pandas-dataframe)
    - [2.1. Creating DataFrame object from Series object, list, dictionary, and NumPy array](#21-creating-dataframe-object-from-series-object--list--dictionary--and-numpy-array)
    - [2.2. Index, values, and cloumns](#22-index--values--and-cloumns)
    - [2.3. Column selection, addition, and deletion](#23-column-selection--addition--and-deletion)
    - [2.4. Row selection, addition, and deletion](#24-row-selection--addition--and-deletion)

<!-- /TOC -->

<br>

# 1. Pandas series
## 1.1. Creating Series object from list and dictionary
* ### from list
    ```
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
    
    <br>

* ### from list with explicit index
    ```
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

    <br>

* ### from dictionary
    ```
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

<br>

## 1.2. Index and values
* ### extract all the index and values
    `s.index` <br>
    `s.values`

* ### select rows by implicit index <br>
    `s[2]` <br>
    `s[2:4]`, inclusive, exclusive <br>
    `s.iloc[2:4]` <br>

* ### select rows by explicit index <br>
    `s['dog']` <br>
    `s['dog':'parrot']`, inclusive, inclusive <br>
    `s.loc['dog':'parrot']` <br>
    
* ### select rows by boolean mask <br>
    `s[s > 2]` <br>
    
<br>

# 2. Pandas DataFrame
## 2.1. Creating DataFrame object from Series object, list, dictionary, and NumPy array
* ### from a dictionary of Series
    ```
    df = pd.DataFrame({'age':s_age, 'color':s_color})
    ```
    **df:**
    |    | age | color |
    |--- | --- | ---   |
    |dog |2    | black |
    |cat |3    | white |
    |parrot|10 | green |

    <br>

* ### from a list 
    ```
    age_list = [2, 3, 10]
    df = pd.DataFrame(age_list)
    ```
    **df:**
    |   |   |
    |---|---|
    |0  |  2|
    |1  |  3|
    |2  | 10|

    <br>

* ### from a list of list
    ```
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

    <br>

* ### from a list of dictionaries
    ```
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

    <br>

* ### from a dictionary of lists
    ```
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

    <br>

* ### assign explicit index from a column
    ```
    df = df2.set_index('type')
    ```
    **df:**
    |    | age | color |
    |--- | --- | ---   |
    |dog |2    | black |
    |cat |3    | white |
    |parrot|10 | green |

<br>

## 2.2. Index, values, and cloumns
* ### Extract index, values, and columns
    `df.index` <br>
    `df.values` <br>
    `df.columns` <br>

<br>

## 2.3. Column selection, addition, and deletion
* ### Select columns
    `df['age']` <br>

* ### Reorganize columns
    `df_reorg = df[['color', 'age']]` <br>

* ### Rename columns
    `df_rename = df.rename(columns = {'age':'Age', 'color':'Color'})`

* ### Create a new column
    `df['age (months)'] = df['age'] * 12` <br>

* ### Delete a column
    `del df['age (months)']` <br>
    alternatively <br>
    `df.pop('age (months)')` <br>
    **df:**
    |    | age | color |
    |--- | --- | ---   |
    |dog |2    | black |
    |cat |3    | white |
    |parrot|10 | green |

<br>

## 2.4. Row selection, addition, and deletion
* ### Select rows by implicit index (integer locaction)
    `df.iloc[0:2]` <br>

* ### Select rows by explicit index (label)
    `df.loc['cat':'parrot']` <br>

* ### Add new rows
    `df_to_add = pd.DataFrame({'age': [1, 4]}, index = ['snake', 'turtle'])` <br>
    `df_append = df.append(df_to_add)` <br>
    **df_append:**
    |    | age | color |
    |--- | --- | ---   |
    |dog |2    | black |
    |cat |3    | white |
    |parrot|10 | green |
    |snake |1  | NaN   |
    |turtle|4  | NaN   |

    <br>

* ### Delete rows by explicit index (label)
    `df_drop = df_append.drop('snake')` <br>
    **df_drop:**
    |    | age | color |
    |--- | --- | ---   |
    |dog |2    | black |
    |cat |3    | white |
    |parrot|10 | green |
    |turtle|4  | NaN   |

<br>


