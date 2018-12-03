# **Python notes**

Notes taken from [Northwestern University Data Science Bootcamp](https://bootcamp.northwestern.edu/data/) and [online materials](https://github.com/amarallab/Introduction-to-Python-Programming-and-Data-Science)

<br>

<!-- TOC -->

- [**Python notes**](#python-notes)
    - [1. Data types](#1-data-types)
        - [1.1. Single value](#11-single-value)
        - [1.2. Collections](#12-collections)
    - [2. Operations on single values](#2-operations-on-single-values)
        - [2.1. Integer and float](#21-integer-and-float)
        - [2.2. Boolean](#22-boolean)
        - [2.3. Built-in methods on string](#23-built-in-methods-on-string)
    - [3. Operations on collections](#3-operations-on-collections)
        - [3.1. List](#31-list)
        - [3.2. Tuple](#32-tuple)
        - [3.3. Set](#33-set)
        - [3.4. Dictionary](#34-dictionary)
    - [4. Flow control](#4-flow-control)
        - [4.1. What if](#41-what-if)
        - [4.2. While loop](#42-while-loop)
        - [4.3. For loop](#43-for-loop)
    - [5. Error handling](#5-error-handling)
        - [5.1. Handling exceptions](#51-handling-exceptions)
        - [5.2. Checking the validity of code](#52-checking-the-validity-of-code)
        - [5.3. Debugging](#53-debugging)
    - [6. File I/O](#6-file-io)
        - [6.1. Reading files](#61-reading-files)
        - [6.2. Writing files](#62-writing-files)
        - [6.3. Encoding](#63-encoding)
    - [7. Standard library](#7-standard-library)
        - [7.1. Documentation](#71-documentation)
        - [7.2. Greatest hits](#72-greatest-hits)
    - [8. Functions](#8-functions)
        - [8.1. Define a function](#81-define-a-function)
        - [8.2. Call the function](#82-call-the-function)
    - [9. Python classes, objects, and methods](#9-python-classes-objects-and-methods)
        - [9.1. Class](#91-class)
        - [9.2. Class with methods](#92-class-with-methods)

<!-- /TOC -->

<br>

## 1. Data types

### 1.1. Single value

- integer
- float
- boolean
- string

### 1.2. Collections

- list
- tuple
- set
- dictionary

Everything about formatting: [documentation](https://pyformat.info/)

| | creation | ordered | mixed data types | elements accessed by | mutable | repeatable |
| :---| :--: | :-----: | :--------------: | :------------------: | :-----: | :--------: |
| **list** | [] |  y   |    y             | index                | y       | y          |
| **tuple** | () | y   |    y             | index                | n       | y          |
| **set** | {} |   n   |    y             | key                  | y       | n          |
| **dictionry** | {} | n |  y             | key                  | y       | n          |

<br>

## 2. Operations on single values

### 2.1. Integer and float

- `//`, truncating division
- `%`, remainder
- `**`, exponentiation

### 2.2. Boolean

- `==`, note assignment vs equality test
- `!=`
- `and`
- `or`
- `not`

### 2.3. Built-in methods on string

- `.capitalize()`, capitalizes the first character <br>
- `.lower()`, makes the entire string lowercase <br>
- `.upper()`, makes the entire string uppercase <br>
- `.title()`, capitalizes every word in a string <br>

- `.strip(' ')`, strip away characters from the right side <br>
- `.strip()`, remove all whitespaces from both sides <br>
- `.lstrip(' ')`, strip away characters starting from the left <br>

- `.split(',')`, split <br>
- `','.join(['a','b'])`, opposite of split <br>

- `.isalpha`, check if all of the characters are alphabetical <br>
- `.isnumeric`, check if the string is a number <br>

- `"Hi my name is {}. My hobby is {}?".format(name, hobby)`
- `f"Hi my name is {name}. My hobby is {hobby}"`, f-string, only available in Python 3.6+
- `"{:.1%}".format(percentage)`, formatting string

    Find substrings inside of strings:
- `.find('the')`, returns index if found, or `-1` if not found
- `.index('the')`, returns index if found, or `ValueError` if not found

<br>

## 3. Operations on collections

### 3.1. List

- #### Add, remove, replace <br>
    ```python
    items = []
    items.append('item1')
    items.extend(['item2', 'item3'])
    items + ['item4']
    items.insert(4, 'item5')  
    ```
    ```python
    items[0] = 'item1_new'
    items.index('item5')
    ```
    ```python
    items.pop(4)
    items.remove('item4')
    del items[-1]
    ```

- #### Zip
    ```python
    zip(names, ages) # zip multiple lists into a list of tuples
    ```

- #### Order
    ```python
    items.reverse() # methods act on the variable directly
    items.sort()
    reversed(items) # functions keep the original variable unchanged
    sorted(items)
    ```

- #### Enumerate
    ```python
    for index, name in enumerate(names): # enumerate creates a list of tuples
        print(f"Name {name} is at index {index}")
    ```

- #### List comprehension
    ```python
    price_strings = ['24', '13', '1']
    price_nums = [int(price) for price in price_strings]
    price_new = [int(price) for price in price_strings if price != '1']
    ```

### 3.2. Tuple

- #### Once created, cannot be readily modified
    ```python
    penny = (60, 'yellow')
    penny + ('amber', )    # add a , so () is different from math operation
    ```

### 3.3. Set

- #### Add, remove, replace
    ```python
    pets = {}
    pets.add('bulldog')
    pets.discard('bulldog')
    ```

- #### Set operations
    ```python
    set1 = {}
    set2 = {}
    set1.intersection(set2) # intersection
    set1.union(set2)        # union
    set1.difference(set2)   # difference set1 - (intersection set1 and set2)
    ```

### 3.4. Dictionary

- #### Add, remove, replace

    ```python
    roster = {}
    roster['Favorite Sport'] = 'Soccer' # add a new item
    roster.update({'Favorite Sports Team': 'S. L. Benfica', 'Favorite Sports Team Mascot': 'Eagle'}) # add multiple new items
    del roster['Favorite Sports Team Mascot']
    print( roster.pop('Favorite Sports Team') )
    ```

- #### Access all items, keys, or values

    ```python
    roster.items() # returns iterators
    roster.keys()
    roster.values()
    ```

<br>

## 4. Flow control

### 4.1. What if

```python
if 'a' in 'abcd':
    ...
elif ...:
    ...
else:
    ...
```

### 4.2. While loop

```python
while ...:
    ...
```

### 4.3. For loop

```python
for i in range(1, 5):
    ...
```

<br>

## 5. Error handling

### 5.1. Handling exceptions

```python
try:
    number = int(number_to_square)
    print("Your number squared is ", number**2)
except:
    print("You didn't enter an integer!")
```

### 5.2. Checking the validity of code

```python
def square(number):
    return square_of_number
assert(square(3) == 9)
```

### 5.3. Debugging

```python
import pdb; pdb.set_trace() # code will run up to this line
```

<br>

## 6. File I/O

### 6.1. Reading files

- #### Generally
    ```python
    file_in = open('Data/ages.csv', 'r')
    lines_str = file_in.read()          # read file into a string
    lines_list = file_in.readlines() # read file into a list of strings
    file_in.close
    ```

- #### Alternatively
    ```python
    with open('Data/ages.csv', 'r') as file_in:
        lines_list = file_in.readlines()
    ```

- #### Reading `csv` file
    
    ```python
    import csv
    with open('Data/ages.csv', newline='') as file_in:
        csvreader = csv.reader(file_in, delimiter=',')
        next(csvreader, None) # skip header
        for row in csvreader:
            print(row)
    ```

- #### Reading `csv` file to dictionary
    
    ```python
    import csv
    with open('names.csv', newline='') as file_in:
        reader = csv.DictReader(file_in)
        for row in reader:
            print(row['first_name']
    ```

- #### Reading `json` file
    `json` stores complex data structures that take on these forms:

  - object (e.g., dictionary)
  - array <br>

    ```python
    import json
    with open('records.json', 'r') as file_in:
        loaded_records = json.load(file_in)
    ```

### 6.2. Writing files

- #### Generally

    ```python
    delimiter = ','
    file_out = open('../Data/TA_ages.csv', 'w')
    for name, age in all_records:
        file_out.write(name + delimiter + str(age) + '\n')    
    file_out.close()
    ```

- #### Alternatively

    ```python
    delimiter = ','
    with open('../Data/TA_ages.csv', 'w') as file_out:
        for name, age in all_records.items():
            file_out.write(name + delimiter + str(age) + '\n')
    file_out.close()
    ```

- #### Writing `csv` file

    ```python
    delimiter = ','
    import csv
    with open(output_path, 'w', newline='') as file_out:
        csvwriter = csv.writer(file_out, delimiter=',')
        csvwriter.writerow(['', '', ''])
    ```

- #### Writing dictionary to `csv` file

    ```python
    import csv
    with open('names.csv', 'w', newline='') as csvfile:
        fieldnames = ['first_name', 'last_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
        # writer.writerow(list_of_dictionary)
    ```

- #### Reading `json` file

    ```python
    import json
    with open('records.json', 'w') as file_out:
        json.dump(all_records, file_out)
    ```

### 6.3. Encoding

- #### Detect file encoding using an encoding dicitonary

    ```python
    # Python has a file containing a dictionary of encoding names and associated aliases
    from encodings.aliases import aliases
    alias_values = set(aliases.values())
    
    for alias in alias_values:
        try:
            df = pd.read_csv('mystery.csv', encoding=alias)
            print(alias)
        except:
            pass
    ```

- #### Detect file encoding using a library

    ```python
    import chardet

    # use the detect method to find the encoding
    # 'rb' means read in the file as binary
    with open("mystery.csv", 'rb') as file:
        print(chardet.detect(file.read()))
    ```

## 7. Standard library

### 7.1. Documentation

- [Brief Tour of the Standard Library](https://docs.python.org/3/tutorial/stdlib.html)
- [The Python Standard Library - Index](https://docs.python.org/3/library/index.html)

### 7.2. Greatest hits

- #### math

- #### random
    `random.random()`, returns a number in the range [0.0, 1.0) <br>
    `random.randint(a, b)`, returns an integer in the range [a, b] <br>
    `random.choice(x)`, randomly returns a value from the sequence x <br>
    `random.sample(x, y)`, randomly returns a sample of length y from the sequence x without replacement

- #### os
    `os.getcwd()`, get current directory <br>
    `os.listdir(directory_name)`, list of files in the directory <br>
    `os.path.join(file_directory, 'file_name.txt')`
    `print(os.path.exists(file_path))`, check if the path exists

- #### glob
    `glob.glob(directory_name + '/*.py')`, return a list of paths matching a pathname pattern

- #### time
    `time.sleep(x)`, pauses for x seconds <br>
    `time.time()`, gets current time in seconds <br>

- #### datetime
    |     |     |
    | --- | --- |
    | `today = datetime.date.today()` <br> `print(today)` <br> `print(today.day)` <br> `print(today.month)` <br> `print(today.year)` | `birthday = datetime.date(1984, 2, 25)` <br> `print(birthday)` <br> `print(birthday.day)` <br> `print(birthday.month)` <br> `print(birthday.year)` |
    |     |     |

    ```python
    raw_time = "Mon May 21 20:50:07 +0000 2018"
    datetime.strptime(raw_time, "%a %b %d %H:%M:%S %z %Y")
    ```
    ```python
    diff_seconds = (converted_timestamps0 - converted_timestamps1).seconds
    ```

    [strftime-and-strptime-behavior](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior) <br>
    [strftime](http://strftime.org/)

- #### copy
    `copy.copy(x)`, shallow copy of x <br>
    `copy.deepcopy(x)`, deep copy of x <br>

- #### operator
    `x.sort(key=operator.itemgetter(2))`, sort based off the 3nd value of each list in x

- #### collections
    `collections.Counter`, counts repeated instances from an iterable

- #### numpy 
    [Package documentation](http://docs.scipy.org/doc/numpy/)

- #### scipy
    [Package documentation](http://docs.scipy.org/doc/scipy/reference/)

<br>

## 8. Functions

### 8.1. Define a function

```python
def function_name(input_var1, input_var2 = "Anna"): 
                # var2 has a default and is optional
    """
    Return all roster filenames in directory
    input:
        input_var - str, Directory that contains the roster files
    output:
        output_var - list, List of roster filenames in directory
    """
    statements
    return output_var1, output_var2 # a tuple of both variables are returned
```

### 8.2. Call the function

```python
output_var1, output_var2 = function_name(input_var1, input_var2) # unpacking
print(function_name.__doc__) # print the docstring
```

<br>


## 9. Python classes, objects, and methods

### 9.1. Class

- #### Define a class

    ```python
    class Dog(): # always capitalize class names

        # Utilize the Python constructor to initialize the object
        def __init__(self, name, color):
            self.name = name
            self.color = color
    ```

- #### Create an instance of a class

    ```python
    dog = Dog('Fido', 'brown')
    ```

- #### Print the object's attributes

    ```python
    print(dog.name)
    print(dog.color)
    ```

### 9.2. Class with methods

- #### Define a class

    ```python
    # Define the Film class
    class Film():

        # A required function to initialize a film object
        def __init__(self, name, length, release_year, language):
            self.name = name
            self.length = length
            self.release_year = release_year
            self.language = language
    ```

- #### Create an object

    ```python
    # An object belonging to the Film class
    star_wars = Film("Star Wars", 121, 1977, "English")
    ```

- #### Define another class with method

    ```python
    # Define the Expert class
    class Expert():
        
        expert_count = 0

        # A required function to initialize the class object
        def __init__(self, name):
            self.name = name
            Expert.expert_count += 1

        # A method that takes another object as its argument
        def boast(self, obj):

            # Print out Expert object's name
            print("Hi. My name is", self.name)
            
            # Print out the name of the Film class object
            print("I know a lot about", obj.name)
            print("It is", obj.length, "minutes long")
            print("It was released in", obj.release_year)
            print("It is in", obj.language)
    ```

- #### Call the method

    ```python
    expert = Expert("Elbert")
    expert.boast(star_wars)
    ```
