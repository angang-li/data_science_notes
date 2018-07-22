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
  - [3. D3](#3-d3)
    - [3.1. D3 in HTML](#31-d3-in-html)
    - [3.2. D3 methods](#32-d3-methods)
    - [3.3. D3 table](#33-d3-table)
    - [3.4. Event listeners](#34-event-listeners)
    - [3.5. This](#35-this)
    - [3.6. Forms](#36-forms)

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

## 3. D3

> D3.js is a JavaScript library for producing dynamic, interactive data visualizations in web browsers.

### 3.1. D3 in HTML

```html
<head>
  ...
  <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/4.7.3/d3.min.js"></script>
</head>
...
<script src="static/js/index.js"></script>
```

### 3.2. D3 methods

#### Select element

- Select one

  ```js
  d3.select(".text1") // class selector
  d3.select("#text1") // id selector
  ```

- Select all

  ```js
  d3.selectAll("li") // tag selector
  ```

- Select an element's child element using `>`

  ```js
  var myLinkAnchor = d3.select(".my-link>a"); // <a href="...">...</a>
  ```

#### Element's text

- Select text

  ```js
  var text1 = d3.select(".text1").text();
  ```

- Modify text

  ```js
  d3.select("#text1").text("Hey, I changed this!");
  ```

#### Element's HTML link

- Capture the HTML of a selection

  ```js
  var myLink = d3.select(".my-link").html(); // myLink: "http://...."
  ```

#### Element's attribute

- Select an attribute

  ```js
  var myLinkAnchorAttribute = myLinkAnchor.attr("href"); // "http://...."
  ```

- Modify an attribute

  ```js
  myLinkAnchor.attr("href", "https://python.org");
  ```

#### Chaining

- Use chaining to join methods

  ```js
  d3.select(".my-link>a").attr("href", "https://nytimes.org").text("Now this is a link to the NYT!!");
  ```

#### Element's style

- Modify style

  ```js
  d3.selectAll("li").style("color", "blue");
  ```

#### Add element

- Append a new list element

  ```js
  var li1 = d3.select("ul").append("li");
  li1.text("A new item has been added!");
  ```

- Define a new image element

  ```js
  d3.select(".giphy-me").html("<img src='https://gph.to/2Krfn0w' alt='giphy'>");
  ```

#### Delete element

- Delete all child elements

  ```js
  d3.select("ul").html("");
  ```

### 3.3. D3 table

- Add table entries dynamically

  ```js
  // Get a reference to the table body
  var tbody = d3.select("tbody");

  // for each object in an array
  data.forEach((weatherReport) => {

      // add a new row
      var row = tbody.append("tr");

      // for each key value pair in an object
      Object.entries(weatherReport).forEach(([key, value]) => {

          // add a new cell
          var cell = row.append("td");
          cell.text(value);

      });
  });
  ```

### 3.4. Event listeners

- Trigger changes when an event happens

  - `button.on("click", function_name)` triggers changes when button is clicked
  - `inputField.on("change", function_name)` triggers changes when new text is entered

  HTML:

  ```html
  ...
  <body>
    <div>
      <button id="click-me">Click Me!</button>
      <input id="input-field" type="text">
    </div>
    <div class="giphy-me"></div>
  </body>
  ...
  ```

  JS:
  
  ```js
  var button = d3.select("#click-me");
  var inputField = d3.select("#input-field");

  // Event handlers are just normal functions that can do anything you want
  button.on("click", function() {
      d3.select(".giphy-me").html("<img src='https://gph.to/2Krfn0w' alt='giphy'>");
      console.log("Hi, a button was clicked!");
      console.log(d3.event.target);
  });

  // Input fields can trigger a change event when new text is entered.
  inputField.on("change", function() {
      var newText = d3.event.target.value;
      console.log(newText);
  });
  ```

### 3.5. This

- A special pointer in JS.

  HTML:

  ```html
  <body>
    <button id="button">Click Me</button>
    <button id="button">Click Me 2</button>
    <ul>
      <li>Item 1</li>
      <li>Item 2</li>
      <li>Item 3</li>
    </ul>
    <script src="https://d3js.org/d3.v4.min.js"></script>
    <script src="app.js"></script>
  </body>
  ```

  JS:

  ```js
  d3.selectAll("button").on("click", function() {
      // `this` will console log the `button` element
      console.log(this); // d3.event.target, target of the event
  });

  d3.selectAll("li").on("click", function() {
      // you can select the element just like any other selection
      var listItem = d3.select(this);
      listItem.style("color", "blue");

      var listItemText = listItem.text();
      console.log(listItemText);
  });
  ```

### 3.6. Forms

- Trigger changes when button is clicked after form entry

  HTML:

  ```html
  <body>
    <div class="container">
      <div class="row">
        <div class="col-md-12">
          <form>
            <div class="form-group">
              <label for="example-form">Enter some text</label>
              <input class="form-control" id="example-form-input" name="example-form" type="text">
            </div>
            <button id="submit" type="submit" class="btn btn-default">Submit</button>
          </form>
        </div>
      </div>
      <div class="row">
        <div class="col-md-12">
          <h1>Form Data:
            <span></span>
          </h1>
        </div>
      </div>
    </div>
  </body>
  <script src="index.js"></script>
  ```

  JS:

  ```js
  // Select the submit button
  var submit = d3.select("#submit");

  submit.on("click", function() {

      // Prevent the page from refreshing
      d3.event.preventDefault();

      // Select the input element and get the raw HTML node
      var inputElement = d3.select("#example-form-input");

      // Get the value property of the input element
      var inputValue = inputElement.property("value");

      console.log(inputValue);

      // Set the span tag in the h1 element to the text that was entered in the form
      d3.select("h1>span").text(inputValue);
  });
  ```