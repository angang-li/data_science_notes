# Python notes

####################
# Data types
    # single value
        integer
        float
        boolean
        string
    # collections
        list
        tuple
        set
        dictionary

# Operations on integer and float
    // # truncating division
    %  # remainder
    ** # exponentiation

# Operators for boolean
    == # note assignment vs equality test
    !=
    and
    or
    not

# Built-in methods on string
    .capitalize() # capitalizes the first character
    .lower() # makes the entire string lowercase
    .upper() # makes the entire string uppercase
    .title() # capitalizes every word in a string

    .strips(' ') # strip away characters from the right side 
    .lstrip(' ') # strip away characters starting from the left

    .split(',') # split

    .isalpha # check if all of the characters are alphabetical
    .isnumeric # check if the string is a number

####################
# Flow control
# What if
    if 'a' in 'abcd':
        ...
    elif ...:
        ...
    else:
        ...

# While loop
    while ...:
        ...

# For loop:
    for i in range(1, 5):
        ...

####################
# Error handling
# Handling exceptions
    try: 
        number = int(number_to_square)
        print("Your number squared is ", number**2)
    except:
        print("You didn't enter an integer!")

# Checking the validity of code
    def square(number):
        return square_of_number
    assert(square(3) == 9)

# Debugging
    import pdb; pdb.set_trace() # code will run up to this line

####################
# List, tuple, and set
        creation  ordered  mixed data types  elements accessed by  mutable  repeatable
list    []        y        y                 index                 y        y
tuple   ()        y        y                 index                 n        y
set     {}        n        y                 key                   y        n

# List
    # add, remove, replace
    items = []
    items.append('item1')
    items.extend(['item2', 'item3'])
    items + ['item4']
    items.insert(4, 'item5')

    items[0] = 'item1_new'
    items.index('item5')

    items.pop(4)
    items.remove('item4')
    del items[-1]

    # order
    items.reverse() # methods act on the variable directly
    items.sort()
    reversed(items) # functions keep the original variable unchanged
    sorted(items)

# Tuple
    # once created, cannot be readily modified
    penny = (60, 'yellow')
    penny + ('amber', ) # , so () is different from math operation

# Set
    # add, remove, replace
    pets = {}
    pets.add('bulldog')
    pets.discard('bulldog')

    # set operations
    set1 = {}
    set2 = {}
    set1.intersection(set2) # intersection
    set1.union(set2)        # union
    set1.difference(set2)   # difference set1 - (intersection set1 and set2)

####################
# File I/O
# Reading files
    # generally
    file_in = open('Data/ages.csv', 'r')
    lines_str = file_in.read() # read file into a string
    lines_list = file_in.readlines() # read file into a list of strings
    file_in.close

    # alternatively
    with open('Data/ages.csv', 'r') as file_in:
        lines_list = file_in.readlines()

    # reading csv files
    import csv
    with open('Data/ages.csv', newline='') as file_in:
        csvreader = csv.reader(file_in, delimiter=',')
        for row in csvreader:
            print(row)

# Writing files
    delimiter = ','
    age_dictionary = zip(names, ages) # zip into a list of tuples
    
    # generally
    file_out = open('../Data/TA_ages.csv', 'w')
    for name, age in age_dictionary:
        file_out.write(name + delimiter + str(age) + '\n')    
    file_out.close()

    # alternatively
    with open('../Data/TA_ages.csv', 'w') as file_out:
        for name, age in age_dictionary.items():
            file_out.write(name + delimiter + str(age) + '\n')
    file_out.close()

    # writing csv files
    with open(output_path, 'w', newline='') as file_out:
        csvwriter = csv.writer(file_out, delimiter=',')
        csvwriter.writerow(['', '', ''])

