# **Python notes**

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
    - [3.4 Dictionary](#34-dictionary)
- [4. Flow control](#4-flow-control)
    - [4.1. What if](#41-what-if)
    - [4.2. While loop](#42-while-loop)
    - [4.3. For loop](#43-for-loop)
- [5. Error handling](#5-error-handling)
    - [5.1. Handling exceptions](#51-handling-exceptions)
    - [5.2. Checking the validity of code](#52-checking-the-validity-of-code)
    - [5.3. Debugging](#53-debugging)
- [6. File I/O](#6-file-i-o)
    - [6.1. Reading files](#61-reading-files)
    - [6.2. Writing files](#62-writing-files)
- [7. Standard library](#7-standard-library)
    - [7.1. Documentation](#71-documentation)
    - [7.2. Greatest hits](#72-greatest-hits)
- [8. Functions](#8-functions)

<!-- /TOC -->

<br>

# 1. Data types
## 1.1. Single value
* integer
* float
* boolean
* string
## 1.2. Collections
* list
* tuple
* set
* dictionary

| | creation | ordered | mixed data types | elements accessed by | mutable | repeatable |
| :---| :--: | :-----: | :--------------: | :------------------: | :-----: | :--------: |
| **list** | [] |  y   |    y             | index                | y       | y          |
| **tuple** | () | y   |    y             | index                | n       | y          |
| **set** | {} |   n   |    y             | key                  | y       | n          |
| **dictionry** | {} | n |  y             | key                  | y       | n          |

<br>

# 2. Operations on single values
## 2.1. Integer and float
* `//`, truncating division
* `%`, remainder
* `**`, exponentiation

## 2.2. Boolean
* `==`, note assignment vs equality test
* `!=`
* `and`
* `or`
* `not`

## 2.3. Built-in methods on string
* `.capitalize()`, capitalizes the first character <br>
* `.lower()`, makes the entire string lowercase <br>
* `.upper()`, makes the entire string uppercase <br>
* `.title()`, capitalizes every word in a string <br>
<br>
* `.strips(' ')`, strip away characters from the right side <br>
* `.lstrip(' ')`, strip away characters starting from the left <br>
<br>
* `.split(',')`, split <br>
* `','.join(['a','b'])`, opposite of split <br>
<br>
* `.isalpha`, check if all of the characters are alphabetical <br>
* `.isnumeric`, check if the string is a number <br>
<br>
* `"Hi my name is {}. My hobby is {}?".format(name, hobby)`
* `f"Hi my name is {name}. My hobby is {hobby}"`, f-string, only available in Python 3.6+
* `"{:.1%}".format(percentage)`, formatting string

<br>

# 3. Operations on collections
## 3.1. List
* ### Add, remove, replace <br>
        items = []
        items.append('item1')
        items.extend(['item2', 'item3'])
        items + ['item4']
        items.insert(4, 'item5')  
    -----
        items[0] = 'item1_new'
        items.index('item5')
    -----
        items.pop(4)
        items.remove('item4')
        del items[-1]

* ### Order
        items.reverse() # methods act on the variable directly
        items.sort()
        reversed(items) # functions keep the original variable unchanged
        sorted(items)

* ### Enumerate
        for index, name in enumerate(names):
            print(f"Name {name} is at index {index}")

* ### List comprehension
        price_strings = ['24', '13', '1']
        price_nums = [int(price) for price in price_strings]
        price_new = [int(price) for price in price_strings if price != '1']

## 3.2. Tuple
* ### Once created, cannot be readily modified
        penny = (60, 'yellow')
        penny + ('amber', )    # add a , so () is different from math operation

## 3.3. Set
* ### Add, remove, replace
        pets = {}
        pets.add('bulldog')
        pets.discard('bulldog')

* ### Set operations
        set1 = {}
        set2 = {}
        set1.intersection(set2) # intersection
        set1.union(set2)        # union
        set1.difference(set2)   # difference set1 - (intersection set1 and set2)

## 3.4 Dictionary
* ### Add, remove, replace
        roster = {}
        roster['Favorite Sport'] = 'Soccer' # add a new item
        roster.update({'Favorite Sports Team': 'S. L. Benfica', 'Favorite Sports Team Mascot': 'Eagle'}) # add multiple new items
        del roster['Favorite Sports Team Mascot']
        print( roster.pop('Favorite Sports Team') )
        
* ### Access all items, keys, or values
        roster.items() # returns iterators
        roster.keys()
        roster.values()

<br>

# 4. Flow control
## 4.1. What if
        if 'a' in 'abcd':
            ...
        elif ...:
            ...
        else:
            ...

## 4.2. While loop
        while ...:
            ...

## 4.3. For loop
        for i in range(1, 5):
            ...

<br>

# 5. Error handling
## 5.1. Handling exceptions
        try: 
            number = int(number_to_square)
            print("Your number squared is ", number**2)
        except:
            print("You didn't enter an integer!")

## 5.2. Checking the validity of code
        def square(number):
            return square_of_number
        assert(square(3) == 9)

## 5.3. Debugging
        import pdb; pdb.set_trace() # code will run up to this line

<br>

# 6. File I/O
## 6.1. Reading files
* ### Generally
        file_in = open('Data/ages.csv', 'r')
        lines_str = file_in.read()          # read file into a string
        lines_list = file_in.readlines() # read file into a list of strings
        file_in.close

* ### Alternatively
        with open('Data/ages.csv', 'r') as file_in:
            lines_list = file_in.readlines()

* ### Reading `csv` file
        import csv
        with open('Data/ages.csv', newline='') as file_in:
            csvreader = csv.reader(file_in, delimiter=',')
            next(csvreader, None) # skip header
            for row in csvreader:
                print(row)

* ### Reading `csv` file to dictionary
        import csv
        with open('names.csv', newline='') as file_in:
            reader = csv.DictReader(file_in)
            for row in reader:
                print(row['first_name']

* ### Reading `json` file
    ### `json` stores complex data structures that take on these forms:
    * ### object (e.g., dictionary)
    * ### array <br> 
    ```
      import json
      with open('records.json', 'r') as file_in:
          loaded_records = json.load(file_in)
    ```

## 6.2. Writing files
```
    delimiter = ','
    age_dictionary = zip(names, ages) # zip into a list of tuples
```

* ### Generally
        file_out = open('../Data/TA_ages.csv', 'w')
        for name, age in age_dictionary:
            file_out.write(name + delimiter + str(age) + '\n')    
        file_out.close()

* ### Alternatively
        with open('../Data/TA_ages.csv', 'w') as file_out:
            for name, age in age_dictionary.items():
                file_out.write(name + delimiter + str(age) + '\n')
        file_out.close()

* ### Writing `csv` file
        import csv
        with open(output_path, 'w', newline='') as file_out:
            csvwriter = csv.writer(file_out, delimiter=',')
            csvwriter.writerow(['', '', ''])

* ### Writing dictionary to `csv` file
        import csv
        with open('names.csv', 'w', newline='') as csvfile:
            fieldnames = ['first_name', 'last_name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
            # writer.writerow(list_of_dictionary)

* ### Reading `json` file
        import json
        with open('records.json', 'w') as file_out:
            json.dump(all_records, file_out)

<br>

# 7. Standard library
## 7.1. Documentation
* [Brief Tour of the Standard Library](https://docs.python.org/3/tutorial/stdlib.html)
* [The Python Standard Library - Index](https://docs.python.org/3/library/index.html)


## 7.2. Greatest hits
* ### math

* ### random
    `random.random()`, returns a number in the range [0.0, 1.0) <br>
    `random.randint(a, b)`, returns an integer in the range [a, b] <br>
    `random.choice(x)`, randomly returns a value from the sequence x <br>
    `random.sample(x, y)`, randomly returns a sample of length y from the sequence x without replacement

* ### os
    `os.getcwd()`, get current directory <br>
    `os.listdir(directory_name)`, list of files in the directory <br>
    `os.path.join(file_directory, 'file_name.txt')`
    `print(os.path.exists(file_path))`, check if the path exists

* ### glob
    `glob.glob(directory_name + '/*.py')`, return a list of paths matching a pathname pattern

* ### time
    `time.sleep(x)`, pauses for x seconds <br>
    `time.time()`, gets current time in seconds <br>

* ### datetime
    |     |     |
    | --- | --- |
    | `today = datetime.date.today()` <br> `print(today)` <br> `print(today.day)` <br> `print(today.month)` <br> `print(today.year)` | `birthday = datetime.date(1984, 2, 25)` <br> `print(birthday)` <br> `print(birthday.day)` <br> `print(birthday.month)` <br> `print(birthday.year)` |
    |     |     |

* ### copy
    `copy.copy(x)`, shallow copy of x <br>
    `copy.deepcopy(x)`, deep copy of x <br>

* ### operator
    `x.sort(key=operator.itemgetter(2))`, sort based off the 3nd value of each list in x

* ### collections
    `collections.Counter`, counts repeated instances from an iterable

* ### numpy 
    [Package documentation](http://docs.scipy.org/doc/numpy/)

* ### scipy
    [Package documentation](http://docs.scipy.org/doc/scipy/reference/)

<br>

# 8. Functions
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

    output_var1, output_var2 = function_name(input_var1, input_var2) # unpacking
    print(function_name.__doc__) # print the docstring

<br>


