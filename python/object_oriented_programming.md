# Object-oriented programming

<!-- TOC -->

- [Object-oriented programming](#object-oriented-programming)
  - [1. Introduction](#1-introduction)
    - [1.1. Vocabulary](#11-vocabulary)
    - [1.2. Procedural vs object-oriented programming](#12-procedural-vs-object-oriented-programming)
    - [1.3. Classes, objects, methods and attributes](#13-classes-objects-methods-and-attributes)
    - [1.4. Module and package](#14-module-and-package)
  - [2. Syntax](#2-syntax)
    - [2.1. Coding a class](#21-coding-a-class)
    - [2.2. Magic methods](#22-magic-methods)
    - [2.3. Inheritance](#23-inheritance)
    - [Unit test](#unit-test)
  - [3. Advanced topics](#3-advanced-topics)
  - [4. Upload a package to Pypi](#4-upload-a-package-to-pypi)
    - [4.1. File structure](#41-file-structure)
    - [4.2. Terminal commands](#42-terminal-commands)
    - [4.3. More resources](#43-more-resources)

<!-- /TOC -->

## 1. Introduction

Objects are defined by characteristics (attributes) and actions (methods).

### 1.1. Vocabulary

- `class` - a blueprint consisting of methods and attributes
- `object` - an instance of a class. It can help to think of objects as something in the real world like a yellow pencil, a small dog, a blue shirt, etc. However, as you'll see later in the lesson, objects can be more abstract.
- `attribute` - a descriptor or characteristic. Examples would be color, length, size, etc. These attributes can take on specific values like blue, 3 inches, large, etc.
- `method` - an action that a class or object could take. Note that a `method` is inside of a class whereas a `function` is outside of a class.
- `OOP` - a commonly used abbreviation for object-oriented programming
- `encapsulation` - one of the fundamental ideas behind object-oriented programming is called encapsulation: you can combine functions and data all into a single entity. In object-oriented programming, this single entity is called a class. Encapsulation allows you to hide implementation details much like how the scikit-learn package hides the implementation of machine learning algorithms.

### 1.2. Procedural vs object-oriented programming

- (+) Object-oriented programming allows you to create large, modular programs that can easily expand over time;
- (+) object-oriented programs hide the implementation from the end-user.

### 1.3. Classes, objects, methods and attributes

- **Class and object:** An object is an instance of a class, and a class is also an object since everything is an object in Python.
- **Method and attribute:** Method can be thought of as a verb, and attribute can be thought of as a noun.

### 1.4. Module and package

- A **module** is a single Python file that contains a collection of functions, classes, and/or global variables. They are modular, meaning that you can reuse these files in different applications.

- A **package** is essentially a collection of modules placed into a directory. a Python package also needs an `__init__.py` file.

## 2. Syntax

### 2.1. Coding a class

- Syntax

  ```python
  class SalesPerson:
      """The SalesPerson class represents an employee in the store
      """

      def __init__(self, first_name, last_name, employee_id, salary):
          """Method for initializing a SalesPerson object
          
          Args: 
              first_name (str)
              last_name (str)
              employee_id (int)
              salary (float)

          Attributes:
              first_name (str): first name of the employee
              last_name (str): last name of the employee
              employee_id (int): identification number of the employee
              salary (float): yearly salary of the employee
              pants_sold (list): a list of pants objects sold by the employee
              total_sales (float): sum of all sales made by the employee
          """
          self.first_name = first_name
          self.last_name = last_name
          self.employee_id = employee_id
          self.salary = salary
          self.pants_sold = []
          self.total_sales = 0

      def sell_pants(self, pants_object):
          """The sell_pants method appends a pants object to the pants_sold attribute

          Args: 
              pants_object (obj): a pants object that was sold

          Returns: None
          """
          self.pants_sold.append(pants_object)
  ```

  - Note that the general object-oriented programming convention is to use methods to access attributes or change attribute values. These methods are called **set and get methods** or setter and getter methods.
  - Note that a **docstring** is a type of comment that describes how a Python module, function, class or method works.

### 2.2. Magic methods

- Magic method lets you overwrite and customize default Python behavior. For example. `__init__(...)` is a magic method.

- Syntax

  ```python
  ...
      def __add__(self, other):
          """Function to add together two Gaussian distributions.
          E.g., gaussian_one + gaussian_two will be valid.
          Args:
              other (Gaussian): Gaussian instance
          Returns:
              Gaussian: Gaussian distribution
          """
          result = Gaussian()
          result.mean = self.mean + other.mean
          result.stdev = math.sqrt(self.stdev ** 2 + other.stdev ** 2)
          return result
          
          
      def __repr__(self):
          """Function to output the characteristics of the Gaussian instance.
          E.g., gaussian_one will report "mean 25, standard deviation 3"
          Args:
              None
          Returns:
              string: characteristics of the Gaussian
          """
          return "mean {}, standard deviation {}".format(self.mean, self.stdev)
  ```

### 2.3. Inheritance

- A child class can **inherit** attributes and behaviour methods from a parent class.

- Syntax

  ```python
  class Clothing:

      def __init__(self, color, size, style, price):
          self.color = color
          self.size = size
          self.style = style
          self.price = price
          
      def calculate_shipping(self, weight, rate):
          return weight * rate
          
  class Blouse(Clothing):
      def __init__(self, color, size, style, price, country_of_origin):
          Clothing.__init__(self, color, size, style, price)
          self.country_of_origin = country_of_origin
      
      def triple_price(self):
          return 3 * self.price
  ```

### Unit test

- Syntax

  ```python
  # Unit tests to check your solution
  import unittest

  class TestClothingClass(unittest.TestCase):
      def setUp(self):
          self.clothing = Clothing('orange', 'M', 'stripes', 35)
          self.blouse = Blouse('blue', 'M', 'luxury', 40, 'Brazil')
          
      def test_initialization(self): 
          self.assertEqual(self.clothing.price, 35, 'incorrect price')
          self.assertEqual(self.blouse.color, 'blue', 'color should be blue')

      def test_calculateshipping(self):
          self.assertEqual(self.blouse.calculate_shipping(.5, 3), .5 * 3,\
          'Clothing shipping calculation not as expected') 
      
  tests = TestClothingClass()

  tests_loaded = unittest.TestLoader().loadTestsFromModule(tests)

  unittest.TextTestRunner().run(tests_loaded)
  ```

## 3. Advanced topics

- [**class methods, instance methods, and static methods**](https://realpython.com/instance-class-and-static-methods-demystified/) - these are different types of methods that can be accessed at the class or object level
- [**class attributes vs instance attributes**](https://www.python-course.eu/python3_class_and_instance_attributes.php) - you can also define attributes at the class level or at the instance level
- [**multiple inheritance, mixins**](https://easyaspython.com/mixins-for-fun-and-profit-cb9962760556) - A class can inherit from multiple parent classes
- [**Python decorators**](https://realpython.com/primer-on-python-decorators/) - Decorators are a short-hand way for using functions inside other functions

## 4. Upload a package to Pypi

Check out an example package [here](example_package)

### 4.1. File structure

The package name must be unique

- A folder with the name of the package that contains:
  - the Python code that makes up the package
  - `README.md`
  - `__init__.py`
  - `license.txt`
  - `setup.cfg`
- `setup.py`

### 4.2. Terminal commands

Should better test in a virtual environment so the development does not mess up with existing installations.

- Prepare for upload

    ```
    cd binomial_package_files
    python setup.py sdist
    pip install twine
    ```

- Upload to Pypi test repository

    ```
    # commands to upload to the pypi test repository
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    pip install --index-url https://test.pypi.org/simple/ dsnd-probability
    ```

- Upload to Pypi repository

    ```
    # command to upload to the pypi repository
    twine upload dist/*
    pip install dsnd-probability
    ```

### 4.3. More resources

- More configuration options for the `setup.py` file: [**tutorial on distributing packages**](https://packaging.python.org/tutorials/packaging-projects/).
- [**Pypi overview**](https://docs.python.org/3/distutils/packageindex.html)
- [**MIT license**](https://opensource.org/licenses/MIT)

