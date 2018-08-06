# JavaScript

Provides interactivity, particularly:

- Dynamic visualization
- IoT & the cloud
- Machine learning in the browser

<!-- TOC -->

- [JavaScript](#javascript)
  - [1. JS fundamentals](#1-js-fundamentals)
    - [1.1. JS in HTML](#11-js-in-html)
    - [1.2. Variables](#12-variables)
    - [1.3. Conditionals](#13-conditionals)
    - [1.4. Arrays](#14-arrays)
    - [1.5. Loops](#15-loops)
    - [1.6. Functions](#16-functions)
    - [1.7. Objects](#17-objects)
  - [2. JS methods and functions](#2-js-methods-and-functions)
    - [2.1. ForEach method](#21-foreach-method)
    - [2.2. Map function](#22-map-function)
    - [2.3. Arrow function](#23-arrow-function)
    - [2.4. Object iteration](#24-object-iteration)
    - [2.5. Filter function](#25-filter-function)
    - [2.6. Other useful methods](#26-other-useful-methods)

<!-- /TOC -->

## 1. JS fundamentals

### 1.1. JS in HTML

```html
<body>
  ...
  <script type="text/javascript">
    console.log("My script is stored within the HTML!")
  </script>
</body>
```

Equivalently,

```html
<body>
  ...
  <!-- src contains the path to the file -->
  <script type="text/javascript" src="app.js"></script>
</body>
```

### 1.2. Variables

- Declare variables

  ```js
  // String
  var name = "...";

  // Integer
  var hourlyWage = 15;

  // Boolean
  var satisfied = true;
  ```

  - `var` declares a variable
  - `let` does not declare a variable, 'ReferenceError'
  - `const` enforces a constant, cannot reassign, but can pop or push

- Convert string to integer

  ```js
  var weeklyHours = "40";
  var weeklyWage = hourlyWage * parseInt(weeklyHours);
  ```

- Print variable to console

  ```js
  console.log("Hello " + name + "!")
  console.log(`Hello ${name}!`)
  ```

### 1.3. Conditionals

- Comparison operators

  `==` equal value <br>
  `===` equal value and equal type <br>
  `!=` not equal <br>
  `!==` not equal value or not equal type <br>
  `>`, `<`, `>=`, `<=`
  
- Logical operators

  `&&` and <br>
  `||` or <br>
  `!` not

- Simple conditional statement

  ```js
  if (x === 1 && y !== 10 && x < y) {
      ...
  }
  ```

- Nested conditional statement

  ```js
  if (x < 10) {
      if (y < 5) {
          ...
      }
      else if (y === 5) {
          ...
      }
      else {
          ...
      }
  }
  ```

### 1.4. Arrays

- Declare an array

  ```js
  var numbers = [1,2,4,7,9];
  numbers[0];
  ```

- Append an element to array

  ```js
  arr.push(new_element);
  ```

- Slicing

  ```js
  numbers.slice(2,4);
  ```

- Sorting

  ```js
  // Sorts descending
  [3, 2, -120].sort(function compareFunction(firstNum, secondNum) {
    // resulting order is (3, 2, -120)
    return secondNum - firstNum;
  });


  // Sorts ascending
  [3, 2, -120].sort(function compareFunction(firstNum, secondNum) {
    // resulting order is (-120, 2, 3)
    return firstNum - secondNum;
  });

  // Arrow Function
  [3, 2, -120].sort((first, second) => first - second);
  ```

  Sort an array of objects

  ```js
  // Sort the data array using the greekSearchResults value
  data.sort(function(a, b) {
    return parseFloat(b.greekSearchResults) - parseFloat(a.greekSearchResults);
  });
  ```

- Split

  ```js
  var sentence = "aaa bbb";
  var words = sentence.split(" ");
  ```

- Join

  ```js
  words.join(" ");
  ```

- Find substring index

  ```js
  var aaa = sentence.indexOf('aaa'); // 0
  var bbb = sentence.indexOf('bbb'); // 4
  ```

- Splice

  ```js
  // At index position zero, remove one element
  var firstSplice = numbers.splice(0, 1);
  ```

### 1.5. Loops

- For loop

  ```js
  // Prototypical use case increments loop counter by one on each iteration
  for (var i = 0; i < 10; i++) {
      console.log("Iteration #", i);
  }
  ```

  Looping through an array

  ```js
  for (var i = 0; i < my_array.length; i+=1) { // equivalent to i++
      console.log(my_array[i]);
  }
  ```

### 1.6. Functions

- Function with no input or output

  ```js
  function printHello() {
      console.log("Hello there!");
  }
  
  printHello();
  ```

- Function with input and output

  ```js
  function addition(a, b) {
      return a + b;
  }

  console.log(addition(44, 50));
  ```

### 1.7. Objects

- Declare an object

  A JavaScript object is similar to a Python dictionary

  ```js
  var movie = {
      name: "Star Wars",
      year: 1977,
      profitable: true,
      sequels: [5, 6, 1, 2, 3, "The Last Jedi"]
  };
  ```

- Value lookup based on key

  ```js
  // JavaScript allows value lookup via dot notation
  movie.name;
  movie.sequels[0];

  // JS also allows value lookup via bracket notation
  movie["name"];
  ```

- Add a key-value pair to an existing object

  ```js
  movie.rating = 8.5;
  ```

- Delete a key-value pair

  ```js
  delete movie.sequels;
  ```

- Check whether a key exists in an object

  ```js
  if ("rating" in movie) {
      console.log("This movie has a rating!");
  }
  ```

- Lookup keys and values

  ```js
  // The keys of the object
  Object.keys(movie)

  // The values of the object
  Object.values(movie)

  // Key-value pairs held in an array
  Object.entries(movie)
  ```

- Conditionally add a value

  ```js
  obj[key] = (obj[key] || 0) + 1;
  ```

## 2. JS methods and functions

### 2.1. ForEach method

`forEach` automatically iterates (loops) through each item and calls the supplied function for that item. This is equivalent to the for loop above.

- Call a supplied function

  ```js
  var students = [
      { name: "Malcolm", score: 80 },
      { name: "Zoe", score: 85 }
  ];

  students.forEach(printName);
  ```

- Call an anonymous function inline

  ```js
  students.forEach(function(name) {
      console.log(name);
  });
  ```

### 2.2. Map function

`map` always returns some variable, whereas `forEach` does not return a callable variable.

- Call a supplied function

  ```js
  var theStagesOfJS = ["confidence", "sadness", "confusion", "realization", "debugging", "satisfaction"];
  
  var mapSimpleArray = theStagesOfJS.map(functionName);
  ```

- Call an anonymous function inline

  ```js
  var mapSimpleArray = theStagesOfJS.map(function(item, index) {
    return item
  });
  ```

### 2.3. Arrow function

An Arrow function (fat arrow `=>`) uses less syntax than the full `map` or `forEach`. Arrow functions allow us to drop the `function` keyword and just show the parameters. [Here](https://medium.freecodecamp.org/when-and-why-you-should-use-es6-arrow-functions-and-when-you-shouldnt-3d851d7f0b26) is a post on when and why to use arrow function.

- Drop the `function` keyword

  ```js
  var mapArrow1 = theStagesOfJS.map((item) => {
    return item;
  });
  ```

- For functions with a single return line, also drop the curly braces

  ```js
  var mapArrow2 = theStagesOfJS.map(item => return "Stage " + item);
  ```

- For functions with a single return line, also drop the `return` keyword

  ```js
  var mapArrow3 = theStagesOfJS.map(item => item);
  ```

- Functions with more than one parameter still need the parenthesis

  ```js
  var mapReturn2 = theStagesOfJS.map((item, index) => `Stage ${index}: ${item}`);
  ```

- Build an array of values from an array of objects

  ```js
  var names = students.map(student => student.name);

  // Use () to avoid confusion of {}
  var new_objs = students.map(student => ({"stu": student.name}));
  ```

### 2.4. Object iteration

- Use `Object.values` and `forEach` to iterate through keys

  ```js
  var userInfo = {
      name: "Eric",
      age: 32,
      location: "North America"
  };

  Object.keys(userInfo).forEach(key => console.log(key));
  ```

- Use `Object.values` and `forEach` to iterate through values

  ```js
  Object.values(userInfo).forEach(value => console.log(value));
  ```

- Use `Object.entries` and `forEach` to iterate through keys and values

  ```js
  Object.entries(userInfo).forEach(([key, value]) => console.log(`Key: ${key} and Value ${value}`));
  ```

- Loop through array of objects then each object

  ```js
  var users = [
      { name: "Eric", age: 32, location: "North America" },
      { name: "Sally", age: 23, location: "Europe" },
      { name: "Cassandra", age: 27, location: "North America" }];

  users.forEach((user) => {
      // Get the entries for each object in the array
      Object.entries(user).forEach(([key, value]) => {
          // Log the key and value
          console.log(`Key: ${key} and Value ${value}`);
      });
  });
  ```

### 2.5. Filter function

- Filter with a custom filtering function

  ```js
  var simpsons = [{name: "Homer", age: 45},
                  {name: "Lisa", age: 8},
                  {name: "Marge", age: 43},
                  {name: "Bart", age: 10},
                  {name: "Maggie", age: 1}];

  // Create a custom filtering function
  function selectYounger(person) {
      return person.age < 30;
  }

  // filter() uses the custom function as its argument
  var youngSimpsons = simpsons.filter(selectYounger);
  ```

- Filter with an anonymous function inline

  ```js
  var youngSimpsons = simpsons.filter(person => person.age < 30);
  ```

### 2.6. Other useful methods

- #### Define a moving average function

  ```js
  // Calculate a rolling average for an array
  function rollingAverage(arr, windowPeriod = 10) {
    // rolling averages array to return
    var averages = [];

    // Loop through all of the data
    for (var i = 0; i < arr.length - windowPeriod; i++) {
      // calculate the average for a window of data
      var sum = 0;
      for (var j = 0; j < windowPeriod; j++) {
        sum += arr[i + j];
      }
      // calculate the average and push it to the averages array
      averages.push(sum / windowPeriod);
    }
    return averages;
  }
  ```

- #### Dynamically find today's date

  ```js
  // Dynamically add the current date to the report header
  var monthNames = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var today = new Date();
  var date = `${monthNames[today.getMonth()]} ${today.getFullYear().toString().substr(2, 2)}`;
  ```

- #### Define a rolling correlation function

  ```js
  // Calculate a rolling correlation for two arrays
  function rollingCorrelation(arr1, arr2, windowPeriod = 10) {
    // correlation array to return
    var corrs = [];
    for (var i = 0; i < arr1.length - windowPeriod; i++) {
      // windows of data to perform correlation on
      var win1 = [];
      var win2 = [];
      for (var j = 0; j < windowPeriod; j++) {
        win1.push(arr1[i + j]);
        win2.push(arr2[i + j]);
      }
      // calculate correlation between two arrays
      corrs.push(ss.sampleCorrelation(win1, win2));
    }
    return corrs;
  }
  ```
