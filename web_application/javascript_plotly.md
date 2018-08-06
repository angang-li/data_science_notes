# Interactive visualization dashboard

highchart, c3

<!-- TOC -->

- [Interactive visualization dashboard](#interactive-visualization-dashboard)
    - [1. Plotly.js](#1-plotlyjs)
        - [1.1. Bar chart](#11-bar-chart)
        - [1.2. Line chart](#12-line-chart)
        - [1.3. Scatter](#13-scatter)
        - [1.4. Pie chart](#14-pie-chart)
        - [1.5. Box plot](#15-box-plot)
        - [1.6. Histogram](#16-histogram)
        - [1.7. Candlestick](#17-candlestick)
    - [2. Plotly dynamics](#2-plotly-dynamics)
        - [2.1. Dynamic graphing with dropdown](#21-dynamic-graphing-with-dropdown)
        - [2.2. Dynamic graphing with form and promise](#22-dynamic-graphing-with-form-and-promise)

<!-- /TOC -->

## 1. Plotly.js

Note:

- Javascript no need "" around keys
- Json needs "" around keys

### 1.1. Bar chart

Note that `Plotly` only picks the last one when there are repeated x values

- #### HTML

  ```html
  <head>
      ...
      <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  </head>
  <body>
      <div id="plot"></div>
      <script src="plots.js"></script>
  </body>
  ```

- #### Create new plot

  ```js
  // Each tracer corresponds to a label
  var trace1 = {
    x: data.map(row => row.greekSearchResults),
    y: data.map(row => row.greekName),
    text: data.map(row => row.greekName),
    name: "Greek",
    type: "bar",
    orientation: "h"
  };

  // data
  var data = [trace1];

  // layout
  var layout = {
    title: "Greek gods search results",
    margin: {
      l: 100,
      r: 100,
      t: 100,
      b: 100
    }
    xaxis: { title: "..."},
    yaxis: { title: "..."},
    barmode: "group" // group traces bars by x
  };

  // Render the plot to the div tag with id "plot"
  Plotly.newPlot("plot", data, layout);

  // Alternatively
  PLOT = document.getElementById('plot');
  Plotly.newPlot(PLOT, data, layout);
  ```

- #### Modify existing plot

  e.g.,

  ```js
  Plotly.relayout("plot", {xaxis: { title: "Drinks"}})
  Plotly.restyle(PLOT, "x", [newx]);
  Plotly.restyle(PLOT, "y", [newy]);
  ```

### 1.2. Line chart

- #### Create new plot

  ```js
  var trace1 = {
    x: ["beer", "wine", "martini", "margarita",
        "ice tea", "rum & coke", "mai tai", "gin & tonic"],
    y: [22.7, 17.1, 9.9, 8.7, 7.2, 6.1, 6.0, 4.6],
    type: "line"
  };

  var data = [trace1];

  var layout = {
    title: "'Bar' Chart",
  };

  Plotly.newPlot("plot", data, layout);
  ```

- #### Additional layout options for date

  ```js
  var layout = {
    title: "'Bar' Chart",
    xaxis: {
      range: [startDate, endDate],
      type: "date"
    },
    yaxis: {
      autorange: true,
      type: "linear"
    }
  };
  ```

### 1.3. Scatter

- #### Create new plot with multiple traces

  ```js
  /**
  * Generates an array of random numbers between 0 and 9
  * @param {integer} n: length of array to generate
  */
  function randomNumbersBetween0and9(n) {
    var randomNumberArray = [];
    for (var i = 0; i < n; i++) {
      randomNumberArray.push(Math.floor(Math.random() * 10));
    }
    return randomNumberArray;
  }

  // Create our first trace
  var trace1 = {
    x: [1, 2, 3, 4, 5],
    y: randomNumbersBetween0and9(5),
    type: "scatter",
    name: "label1",
    mode: "markers",
    // mode: "lines",
    marker: {
      color: "#2077b4",
      symbol: "hexagram"
    },
    // line: {
    //   color: "#17BECF"
    // }
  };

  // Create our second trace
  var trace2 = {
    x: [1, 2, 3, 4, 5],
    y: randomNumbersBetween0and9(5),
    type: "scatter",
    name: "label2",
  };

  // The data array consists of both traces
  var data = [trace1, trace2];

  // Note that we omitted the layout object this time
  // This will use default parameters for the layout
  Plotly.newPlot("plot", data);
  ```

### 1.4. Pie chart

- #### Create new plot

  ```js
  var trace1 = {
    labels: ["beer", "wine", "martini", "margarita",
        "ice tea", "rum & coke", "mai tai", "gin & tonic"],
    values: [22.7, 17.1, 9.9, 8.7, 7.2, 6.1, 6.0, 4.6],
    type: 'pie'
  };

  var data = [trace1];

  var layout = {
    title: "'Bar' Chart",
  };

  Plotly.newPlot("plot", data, layout);
  ```

### 1.5. Box plot

- #### Create new plot

  ```js
  // Create two arrays, each of which will hold data for a different trace
  var y0 = [];
  var y1 = [];

  // Fill each of the above arrays with data
  for (var i = 0; i < 50; i++) {
    y0.push(Math.random());
    y1.push(Math.random() + 1);
  }

  // Create a trace object with the data in `y0`
  var trace1 = {
    y: y0,
    type: "box"
  };

  // Create a trace object with the data in `y1`
  var trace2 = {
    y: y1,
    type: "box"
  };

  // Create a data array with the above two traces
  var data = [trace1, trace2];

  // Use `layout` to define a title
  var layout = {
    title: "Basic Box Plot"
  };

  // Render the plot to the `plot1` div
  Plotly.newPlot("plot1", data, layout);
  ```

- #### Use `boxpoints: all` to render a scatter plot besides the box/whisker diagram

  ```js
  var traceB1 = {
    y: y0,
    boxpoints: "all",
    type: "box"
  };

  var traceB2 = {
    y: y1,
    boxpoints: "all",
    type: "box"
  };

  // Use `layoutB` to set a title
  var layoutB = {
    title: "Box Plot with Points"
  };

  // Create a `dataB` array
  var dataB = [traceB1, traceB2];

  // Render the plot to the `plot2` div
  Plotly.newPlot("plot2", dataB, layoutB);
  ```

### 1.6. Histogram

- #### Create new plot

  ```js
  var x = [];
  for (var i = 0; i < 500; i++) {
    x[i] = Math.random();
  }
  
  var trace2 = {
    x: x,
    type: "histogram",
    autobinx: false,
    xbins: {
      start: 0,
      end: 1,
      size: 0.01
    }
  };

  var data2 = [trace2];

  var layout2 = {
    title: "Histogram with Bin Size 0.01",
    bargap: 0.05
  };

  Plotly.newPlot("plot2", data2, layout2);
  ```

### 1.7. Candlestick

`Candlestick` produces more sophisticated interactive chart

- Create new plot

  ```js
  var trace1 = {
    type: "scatter",
    mode: "lines",
    name: name,
    x: dates,
    y: closingPrices,
    line: {
      color: "#17BECF"
    }
  };
  
  // Candlestick Trace
  var trace2 = {
    type: "candlestick",
    x: dates,
    high: highPrices,
    low: lowPrices,
    open: openingPrices,
    close: closingPrices
  };

  var data = [trace1, trace2];

  var layout = {
    title: `${stock} closing prices`,
    xaxis: {
      range: [startDate, endDate],
      type: "date"
    },
    yaxis: {
      autorange: true,
      type: "linear"
    }
  };

  Plotly.newPlot("plot", data, layout);
  ```

## 2. Plotly dynamics

### 2.1. Dynamic graphing with dropdown

- HTML

  ```html
  <head>
    ...
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  </head>

  <body>
    <div id="plot"></div>
    <script src="plots.js"></script>
    <select id="selDataset" onchange="getData(this.value)">
      <option value="dataset1">DataSet1</option>
      <option value="dataset2">DataSet2</option>
      <option value="dataset3">DataSet3</option>
    </select>
  </body>
  ```

- plots.js

  ```js
  // Initialize plot
  function init() {
    data = [{
      x: [1, 2, 3, 4, 5],
      y: [1, 2, 4, 8, 16] }];
    var LINE = document.getElementById("plot");
    Plotly.plot(LINE, data);
  }

  // Update plot
  function updatePlotly(newx, newy) {
    var LINE = document.getElementById("plot");

    // Note the extra brackets around 'newx' and 'newy'
    Plotly.restyle(LINE, "x", [newx]);
    Plotly.restyle(LINE, "y", [newy]);
  }

  // Update based on dropdown value
  function getData(dataset) {

    // Initialize empty arrays to contain our axes
    var x = [];
    var y = [];

    // Fill the x and y arrays as a function of the selected dataset
    switch (dataset) {
    case "dataset1":
      x = [1, 2, 3, 4, 5];
      y = [0.1, 0.2, 0.3, 0.4, 0.5];
      break;
    case "dataset2":
      x = [10, 20, 30, 40, 50];
      y = [1, 10, 100, 1000, 10000];
      break;
    case "dataset3":
      x = [100, 200, 300, 400, 500];
      y = [10, 100, 50, 10, 0];
      break;
    default:
      x = [1, 2, 3, 4, 5];
      y = [1, 2, 3, 4, 5];
      break;
    }

    updatePlotly(x, y);
  }

  init();
  ```

### 2.2. Dynamic graphing with form and promise

- HTML

  ```html
  <!DOCTYPE html>
  <html lang="en">

  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Financial Plot</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/5.5.0/d3.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  </head>
  
  <body>
    <!-- Form -->
    <form class="form-inline">
      <div class="form-group">
        <label for="stockInput">Stock</label>
        <input type="text" class="form-control" id="stockInput" placeholder="Enter a stock">
      </div>
      <button id="submit" type="submit" class="btn btn-default">Plot Stock</button>
    </form>
    <!-- Plot -->
    <div id="plot"></div>
    <script src="plots.js"></script>
  </body>

  </html>
  ```

- JS function to unpack stock json data

  ```js
  /**
  * Helper function to select stock data
  * Returns an array of values
  * @param {array} rows
  * @param {integer} index
  * index 0 - Date
  * index 1 - Open
  * index 2 - High
  * index 3 - Low
  * index 4 - Volume
  */
  function unpack(rows, index) {
    return rows.map(function(row) {
      return row[index];
    });
  }
  ```

- JS function to submit button handler

  Learn more about D3 selection [here](https://github.com/d3/d3-selection)

  ```js
  function handleSubmit() {
    // Prevent the page from refreshing
    d3.event.preventDefault();

    // Select the input value from the form
    var stock = d3.select("#stockInput").node().value;
    console.log(stock);

    // clear the input value
    d3.select("#stockInput").node().value = "";

    // Build the plot with the new stock
    buildPlot(stock);
  }
  ```

- JS function build plot with promise

  ```js
  var apiKey = "ppvzoohSZsStRq8whQr9";
  var url =
    `https://www.quandl.com/api/v3/datasets/WIKI/AMZN.json?start_date=2016-10-01&end_date=2017-10-01&api_key=${apiKey}`;

  function buildPlot() {
    d3.json(url).then(function(data) {

      // Grab values from the data json object to build the plots
      var name = data.dataset.name;
      var stock = data.dataset.dataset_code;
      var startDate = data.dataset.start_date;
      var endDate = data.dataset.end_date;
      var dates = unpack(data.dataset.data, 0);
      var closingPrices = unpack(data.dataset.data, 1);

      var trace1 = {
        type: "scatter",
        x: dates,
        y: closingPrices,
      };
      var data = [trace1];
      var layout = {
        title: `${stock} closing prices`,
      };

      Plotly.newPlot("plot", data, layout);

    });
  }

  buildPlot();
  ```

- JS event listener for submit button

  ```js
  d3.select("#submit").on("click", handleSubmit);
  ```
