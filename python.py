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

