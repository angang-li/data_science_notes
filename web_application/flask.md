# Flask

[Flask](http://flask.pocoo.org/):
a micro web framework written in Python <br>
[Django](https://www.djangoproject.com/): 
a free and open-source web framework, written in Python

<!-- TOC -->

- [Flask](#flask)
    - [1. API endpoint for jasonified data](#1-api-endpoint-for-jasonified-data)
    - [2. Database backed API](#2-database-backed-api)
    - [3. Render templates](#3-render-templates)
        - [3.1. Python script](#31-python-script)
        - [3.2. HTML template](#32-html-template)
    - [4. Render database backed templates with flask and pymongo](#4-render-database-backed-templates-with-flask-and-pymongo)
        - [4.1. Python data script](#41-python-data-script)
        - [4.2. Python app script](#42-python-app-script)
        - [4.3. HTML template](#43-html-template)
    - [5. Scrape webpage and render database backed templates with flask, pymongo, and beautiful soup](#5-scrape-webpage-and-render-database-backed-templates-with-flask-pymongo-and-beautiful-soup)
        - [5.1. Python scraping script](#51-python-scraping-script)
        - [5.2. Python app script](#52-python-app-script)
        - [5.3. HTML template](#53-html-template)
    - [6. Render database-backed template with dropdown and display plot with plotly](#6-render-database-backed-template-with-dropdown-and-display-plot-with-plotly)
        - [6.1. Python app script](#61-python-app-script)
        - [6.2. HTML template with dropdown and plot](#62-html-template-with-dropdown-and-plot)
        - [6.3. JS with event listener and plotly](#63-js-with-event-listener-and-plotly)
    - [7. Render database-backed template with form post](#7-render-database-backed-template-with-form-post)
        - [7.1. Python app script](#71-python-app-script)
        - [7.2. Render HTML template with form post](#72-render-html-template-with-form-post)
    - [8. Render database-backed template with form post and display plot with plotly](#8-render-database-backed-template-with-form-post-and-display-plot-with-plotly)
        - [8.1. Python app script](#81-python-app-script)
        - [8.2. Render main HTML template with plot and link to form](#82-render-main-html-template-with-plot-and-link-to-form)
        - [8.3. Render HTML form](#83-render-html-form)
        - [8.4. JS to build plot](#84-js-to-build-plot)

<!-- /TOC -->

## 1. API endpoint for jasonified data

* Use Flask to create and run a server
* Define endpoints using Flask's @app.route decorator
* Execute database queries on behalf of the client
* Return JSONified query results from API endpoints

```python
from flask import Flask, jsonify

justice_league_members = [
    {"superhero": "Aquaman", "real_name": "Arthur Curry"},
    {"superhero": "Batman", "real_name": "Bruce Wayne"},
    {"superhero": "Cyborg", "real_name": "Victor Stone"},
    {"superhero": "Flash", "real_name": "Barry Allen"},
    {"superhero": "Green Lantern", "real_name": "Hal Jordan"},
    {"superhero": "Superman", "real_name": "Clark Kent/Kal-El"},
    {"superhero": "Wonder Woman", "real_name": "Princess Diana"}
]

# Flask Setup
app = Flask(__name__)


# Flask Routes
@app.route("/api/v1.0/justice-league")
def justice_league():
    """Return the justice league data as json"""

    return jsonify(justice_league_members)


@app.route("/")
def welcome():
    return (
        f"Welcome to the Justice League API!<br/>"
        f"Available Routes:<br/>"
        f"/api/v1.0/justice-league<br/>"
        f"/api/v1.0/justice-league/Arthur%20Curry<br/>"
        f"/api/v1.0/justice-league/Bruce%20Wayne<br/>"
        f"/api/v1.0/justice-league/Victor%20Stone<br/>"
        f"/api/v1.0/justice-league/Barry%20Allen<br/>"
        f"/api/v1.0/justice-league/Hal%20Jordan<br/>"
        f"/api/v1.0/justice-league/Clark%20Kent/Kal-El<br/>"
        f"/api/v1.0/justice-league/Princess%20Diana"
    )


@app.route("/api/v1.0/justice-league/<real_name>")
def justice_league_character(real_name):
    """Fetch the Justice League character whose real_name matches
       the path variable supplied by the user, or a 404 if not."""

    canonicalized = real_name.replace(" ", "").lower()
    for character in justice_league_members:
        search_term = character["real_name"].replace(" ", "").lower()

        if search_term == canonicalized:
            return jsonify(character)

    return jsonify({"error": f"Character with real_name {real_name} not found."}), 404


if __name__ == "__main__":
    app.run(debug=True)
```

## 2. Database backed API

```python
import datetime as dt
import numpy as np
import pandas as pd

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func

from flask import Flask, jsonify


# Database Setup
engine = create_engine("sqlite:///titanic.sqlite")

Base = automap_base() # reflect an existing database into a new model
Base.prepare(engine, reflect=True) # reflect the tables

Passenger = Base.classes.passenger # save reference to the table
session = Session(engine) # create our session (link) from Python to the DB

# Flask Setup
app = Flask(__name__)


# Flask Routes
@app.route("/")
def welcome():
    """List all available api routes."""
    return (
        f"Available Routes:<br/>"
        f"/api/v1.0/names<br/>"
        f"/api/v1.0/passengers"
    )


@app.route("/api/v1.0/names")
def names():
    """Return a list of all passenger names"""
    # Query all passengers
    results = session.query(Passenger.name).all()

    # Convert list of tuples into normal list
    all_names = list(np.ravel(results))

    return jsonify(all_names)


@app.route("/api/v1.0/passengers")
def passengers():
    """Return a list of passenger data including the name, age, and sex of each passenger"""
    # Query all passengers
    results = session.query(Passenger).all()

    # Create a dictionary from the row data and append to a list of all_passengers
    all_passengers = []
    for passenger in results:
        passenger_dict = {}
        passenger_dict["name"] = passenger.name
        passenger_dict["age"] = passenger.age
        passenger_dict["sex"] = passenger.sex
        all_passengers.append(passenger_dict)

    return jsonify(all_passengers)


if __name__ == '__main__':
    app.run(debug=True)
```

## 3. Render templates

Files

- [app.py](flask/app.py)
- [templates/index.html](flask/templates/index.html)

### 3.1. Python script

```python
# import necessary libraries
from flask import Flask, render_template

# create instance of Flask app
app = Flask(__name__)

# List of dictionaries
dogs = [{"name": "Fido", "type": "Lab"},
        {"name": "Rex", "type": "Collie"},
        {"name": "Suzzy", "type": "Terrier"},
        {"name": "Tomato", "type": "Retriever"}]


# create route that renders index.html template
@app.route("/")
def index():

    return render_template("index.html", dogs=dogs)


if __name__ == "__main__":
    app.run(debug=True)
```

### 3.2. HTML template

```html
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="ie=edge">
  <title>Animal Adoption!</title>
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
</head>

<body>
  <div class="container text-center">
    <h1 class="jumbotron">Dogs up for Adoption!</h1>
    <div>
      <ul style="list-style: none;">

        <!-- Loop through the dictionary -->

        {% for dog in dogs %}
          <li style="color:blue">{{ dog.name }} who is a {{ dog.type }}</li>
        {% endfor %}

      </ul>
    </div>
  </div>
</body>

</html>
```

## 4. Render database backed templates with flask and pymongo

Files

- [data.py](flask/flask_database/data.py)
- [app.py](flask/flask_database/app.py)
- [templates/index.html](flask/flask_database/templates/index.html)

### 4.1. Python data script

```python
import pymongo

# Setup connection to mongodb
conn = "mongodb://localhost:27017"
client = pymongo.MongoClient(conn)

# Select database and collection to use
db = client.store_inventory
collection = db.produce

db.collection.insert_many(
    [
        {
            "type": "apples",
            "cost": .23,
            "stock": 333
        },
        {
            "type": "oranges",
            "cost": .45,
            "stock": 30
        },
        {
            "type": "kiwi",
            "cost": .10,
            "stock": 1000
        },
        {
            "type": "mango",
            "cost": 1.30,
            "stock": 20
        },
        {
            "type": "berries",
            "cost": 2.99,
            "stock": 99
        }
    ]
)

print("Data Uploaded!")
```

### 4.2. Python app script

```python
from flask import Flask, render_template
import pymongo

app = Flask(__name__)

# setup mongo connection
conn = "mongodb://localhost:27017"
client = pymongo.MongoClient(conn)

# connect to mongo db and collection
db = client.store_inventory
collection = db.produce


@app.route("/")
def index():
    # write a statement that finds all the items in the db and sets it to a variable
    inventory = list(db.collection.find())
    print(inventory)

    # render an index.html template and pass it the data you retrieved from the database
    return render_template("index.html", inventory=inventory)


if __name__ == "__main__":
    app.run(debug=True)
```

### 4.3. HTML template

```html
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="ie=edge">
  <title>Stock!</title>
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
</head>

<body>
    <div class="container text-center">
      <h1 class="jumbotron">Inventory</h1>
      <div>
        <!-- write a for loop that lists out your data and accesses the values using dot notation. -->
          {% for item in inventory %}
          <div class="col-lg-4">
              <div class="card" style="width: 20rem;">
                  <div class="card-body">
                    <h4 class="card-title">{{ item.type }}</h4>
                    <h6 class="card-subtitle mb-2 text-muted">Cost: {{ item.cost }}</h6>
                    <p class="card-text">Potential Revenue: {{ item.cost * item.stock }}</p>
                  </div>
                </div>
            </div>
            {% endfor %}

      </div>
    </div>
  </body>

</html>
```

## 5. Scrape webpage and render database backed templates with flask, pymongo, and beautiful soup

Files

- [scraping.py](flask/flask_database_scrape/scraping.py)
- [app.py](flask/flask_database_scrape/app.py)
- [templates/index.html](flask/flask_database_scrape/templates/index.html)

### 5.1. Python scraping script

```python
import time
from splinter import Browser
from bs4 import BeautifulSoup
from datetime import datetime


# Initialize browser
def init_browser():
    # @NOTE: Replace the path with your actual path to the chromedriver
    executable_path = {"executable_path": "/usr/local/bin/chromedriver"}
    return Browser("chrome", **executable_path, headless=False)


# Function to scrape for weather in Cost Rica
def scrape_weather():

    # Initialize browser
    browser = init_browser()

    # Visit the Costa Rica climate site
    url = "https://weather-and-climate.com/average-monthly-Rainfall-Temperature-Sunshine-fahrenheit,san-jose,Costa-Rica"
    browser.visit(url)

    # Scrape page into soup
    html = browser.html
    soup = BeautifulSoup(html, 'html.parser')

    # Find today's forecast
    forecast_today = soup.find("div", class_="weather-forecasts todays-weather forecast")
    forecast_today

    # Get the max temp
    max_temp = forecast_today.find("span", class_="temp-max").text

    # Print the min temp
    min_temp = forecast_today.find("span", class_="temp-min").text

    # Get current time stamp
    time_stamp = str(datetime.now())

    # Store in dictionary
    weather = {
        "time": time_stamp,
        "name": "Costa Rica",
        "min_temp": min_temp,
        "max_temp": max_temp
    }

    # Return results
    return weather
```

### 5.2. Python app script

```python
# import necessary libraries
from flask import Flask, render_template, redirect
from flask_pymongo import PyMongo
import scraping

# create instance of Flask app
app = Flask(__name__)

# Use flask_pymongo to set up mongo connection
mongo = PyMongo(app)


# create route that renders index.html template and finds documents from mongo
@app.route("/")
def home():

    # Find data
    forecasts = mongo.db.collection.find()

    # return template and data
    return render_template("index.html", forecasts=forecasts)


# Route that will trigger scrape functions
@app.route("/scrape")
def scrape():

    # Run scraped functions
    weather = scraping.scrape_weather()

    # Store results into a dictionary
    forecast = {
        "time": weather["time"],
        "location": weather["name"],
        "min_temp": weather["min_temp"],
        "max_temp": weather["max_temp"]
    }

    # Insert forecast into database
    mongo.db.collection.insert_one(forecast)

    # Redirect back to home page
    return redirect("http://localhost:5000/", code=302)


if __name__ == "__main__":
    app.run(debug=True)
```

### 5.3. HTML template

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
</head>

<body>

    <div class="container">
        <div class="jumbotron">
            <h1>Costa Rica Weather</h1>
            <p><a class="btn btn-primary btn-lg" href="/scrape" role="button">Get Forecast!</a></p>

        </div>

            {% for forecast in forecasts%}
                <div class="col-lg-4">
                    <div class="card" style="width: 20rem;">
                        <div class="card-body">
                            <h4 class="card-title">Costa Rica</h4>
                            <h6 class="card-subtitle mb-2 text-muted">{{ forecast.time }} - {{ forecast.name }}: {{ forecast.min_temp }} | {{forecast.max_temp}}</h6>
                        </div>
                    </div>
                </div>
            {% endfor %}

    </div>

</body>

</html>
```

## 6. Render database-backed template with dropdown and display plot with plotly

Files

- [app.py](flask/flask_plotly/app.py)
- [templates/index.html](flask/flask_plotly/templates/index.html)
- [static/js/app.js](flask/flask_plotly/static/js/app.js)
- [db/emoji.sqlite](flask/flask_plotly/db/emoji.sqlite)

### 6.1. Python app script

#### 6.1.1. Flask and database setup

```python
import pandas as pd
from flask import (
    Flask,
    render_template,
    jsonify)
from flask_sqlalchemy import SQLAlchemy

# Flask Setup
app = Flask(__name__)

# Database Setup
# The database URI
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///db/emoji.sqlite"
db = SQLAlchemy(app)

class Emoji(db.Model):
    __tablename__ = 'emoji'

    id = db.Column(db.Integer, primary_key=True)
    emoji_char = db.Column(db.String)
    emoji_id = db.Column(db.String)
    name = db.Column(db.String)
    score = db.Column(db.Integer)

    def __repr__(self):
        return '<Emoji %r>' % (self.name)
```

#### 6.1.2. Create database table before any request

```python
@app.before_first_request
def setup():
    # Recreate database each time for demo
    # db.drop_all()
    db.create_all()
```

#### 6.1.3. Flask Routes

```python
@app.route("/")
def home():
    """Render Home Page."""
    return render_template("index.html")


@app.route("/emoji_char")
def emoji_char_data():
    """Return emoji score and emoji char"""

    # query for the top 10 emoji data
    results = db.session.query(Emoji.emoji_char, Emoji.score).\
        order_by(Emoji.score.desc()).\
        limit(10).all()

    # Another query syntax
    # sel = [func.strftime("%Y", Bigfoot.timestamp), func.count(Bigfoot.timestamp)]
    # results = db.session.query(*sel).\
    #     group_by(func.strftime("%Y", Bigfoot.timestamp)).all()

    # Select the top 10 query results
    emoji_char = [result[0] for result in results]
    scores = [int(result[1]) for result in results]

    # Generate the plot trace
    plot_trace = {
        "x": emoji_char,
        "y": scores,
        "type": "bar"
    }
    return jsonify(plot_trace)


@app.route("/emoji_id")
def emoji_id_data():
    """Return emoji score and emoji id"""

    # query for the emoji data using pandas
    query_statement = db.session.query(Emoji).\
        order_by(Emoji.score.desc()).\
        limit(10).statement
    df = pd.read_sql_query(query_statement, db.session.bind)

    # Format the data for Plotly
    plot_trace = {
        "x": df["emoji_id"].values.tolist(),
        "y": df["score"].values.tolist(),
        "type": "bar"
    }
    return jsonify(plot_trace)


@app.route("/emoji_name")
def emoji_name_data():
    """Return emoji score and emoji name"""

    # query for the top 10 emoji data
    results = db.session.query(Emoji.name, Emoji.score).\
        order_by(Emoji.score.desc()).\
        limit(10).all()
    df = pd.DataFrame(results, columns=['name', 'score'])

    # Format the data for Plotly
    plot_trace = {
        "x": df["name"].values.tolist(),
        "y": df["score"].values.tolist(),
        "type": "bar"
    }
    return jsonify(plot_trace)


if __name__ == '__main__':
    app.run(debug=True)
```

### 6.2. HTML template with dropdown and plot

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Flask Plotlyjs Example</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/5.5.0/d3.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="bar"></div>
    <select id="selDataset" onchange="getData(this.value)">
        <option value="emoji_char">Emoji Char</option>
        <option value="emoji_id">Emoji ID</option>
        <option value="emoji_name">Emoji Name</option>
    </select>

    <script src="{{ url_for('static', filename='js/app.js') }}"></script>

</body>
</html>
```

### 6.3. JS with event listener and plotly

```js
// Plot the default route once the page loads
const defaultURL = "/emoji_char";
d3.json(defaultURL).then(function(data) {
  var data = [data];
  var layout = { margin: { t: 30, b: 100 } };
  Plotly.plot("bar", data, layout);
});

// Update the plot with new data
function updatePlotly(newdata) {
  Plotly.restyle("bar", "x", [newdata.x]);
  Plotly.restyle("bar", "y", [newdata.y]);
}

// Get new data whenever the dropdown selection changes
function getData(route) {
  console.log(route);
  d3.json(`/${route}`).then(function(data) {
    console.log("newdata", data);
    updatePlotly(data);
  });
}
```

## 7. Render database-backed template with form post

Files

- [app.py](flask/flask_form/app.py)
- [templates/form.html](flask/flask_form/templates/form.html)
- [db/db.sqlite](flask/flask_form/db/db.sqlite)

### 7.1. Python app script

#### 7.1.1. Flask and database setup

```python
# import necessary libraries
from flask import (
    Flask,
    render_template,
    jsonify,
    request)
from flask_sqlalchemy import SQLAlchemy

# Flask Setup
app = Flask(__name__)

# Database Setup
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///db.sqlite"
db = SQLAlchemy(app)

class Pet(db.Model):
    __tablename__ = 'pets'

    id = db.Column(db.Integer, primary_key=True)
    nickname = db.Column(db.String(64))
    age = db.Column(db.Integer)

    def __repr__(self):
        return '<Pet %r>' % (self.nickname)
```

#### 7.1.2. Create database table before any request

```python
@app.before_first_request
def setup():
    # Recreate database each time for demo
    db.drop_all()
    db.create_all()
```

#### 7.1.3. Flask Routes with form post

```python
@app.route("/send", methods=["GET", "POST"])
def send():
    if request.method == "POST":
        nickname = request.form["nickname"]
        age = request.form["age"]

        pet = Pet(nickname=nickname, age=age)
        db.session.add(pet)
        db.session.commit()

        return "Thanks for the form data!"

    return render_template("form.html")


@app.route("/pets")
def list_pets():
    results = db.session.query(Pet.nickname, Pet.age).all()

    pets = []
    for result in results:
        pets.append({
            "nickname": result[0],
            "age": result[1]
        })
    return jsonify(pets)


@app.route("/")
def home():
    return "Welcome!"


if __name__ == "__main__":
    app.run()
```

### 7.2. Render HTML template with form post

```html
<form method="POST" action="/send">

    <label for="inputName">Name</label>
    <input type="text" id="inputName" name="nickname">

    <label for="inputAge">Age</label>
    <input type="text" id="inputAge" name="age">

    <input type="submit" value="submit">
</form>
```

## 8. Render database-backed template with form post and display plot with plotly

Files

- [app.py](flask/flask_form_plotly/app.py)
- [templates/index.html](flask/flask_form_plotly/templates/index.html)
- [templates/form.html](flask/flask_form_plotly/templates/form.html)
- [static/js/app.js](flask/flask_form_plotly/static/js/app.js)
- [static/css/style.css](flask/flask_form_plotly/static/css/style.css)
- [db/pets.sqlite](flask/flask_form_plotly/db/pets.sqlite)

### 8.1. Python app script

#### 8.1.1. Flask and database setup

```python
# import necessary libraries
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect)

from flask_sqlalchemy import SQLAlchemy

# Flask Setup
app = Flask(__name__)

# Database Setup
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///db/pets.sqlite"
db = SQLAlchemy(app)

class Pet(db.Model):
    __tablename__ = 'pets'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64))
    lat = db.Column(db.Float)
    lon = db.Column(db.Float)

    def __repr__(self):
        return '<Pet %r>' % (self.name)
```

#### 8.1.2. Create database table before any request

```python
@app.before_first_request
def setup():
    # Recreate database each time for demo
    # db.drop_all()
    db.create_all()
```

#### 8.1.3. Flask Routes with form post and plot data

```python
# create route that renders index.html template
@app.route("/")
def home():
    return render_template("index.html")


# Query the database and send the jsonified results
@app.route("/send", methods=["GET", "POST"])
def send():
    if request.method == "POST":
        name = request.form["petName"]
        lat = request.form["petLat"]
        lon = request.form["petLon"]

        pet = Pet(name=name, lat=lat, lon=lon)
        db.session.add(pet)
        db.session.commit()
        return redirect("/", code=302)

    return render_template("form.html")


@app.route("/api/pals")
def pals():
    results = db.session.query(Pet.name, Pet.lat, Pet.lon).all()

    hover_text = [result[0] for result in results]
    lat = [result[1] for result in results]
    lon = [result[2] for result in results]

    pet_data = [{
        "type": "scattergeo",
        "locationmode": "USA-states",
        "lat": lat,
        "lon": lon,
        "text": hover_text,
        "hoverinfo": "text",
        "marker": {
            "size": 50,
            "line": {
                "color": "rgb(8,8,8)",
                "width": 1
            },
        }
    }]

    return jsonify(pet_data)


if __name__ == "__main__":
    app.run()
```

### 8.2. Render main HTML template with plot and link to form

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Pet Pals!</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>

<body>
    <div class="container">
        <div class="jumbotron" style="text-align: center">
            <h1>Pet Pal</h1>
            <p>Find your pet a pal!</p>
        </div>

        <div class="row">
            <div class="col-md-12">
                <h2>Look at all of the current pet pals!</h2>
                <div id="plot"></div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-12">
                <h2><a href="/send">Add your pet here!</a></h2>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/5.5.0/d3.min.js"></script>

    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="{{ url_for('static', filename='js/app.js') }}"></script>
</body>

</html>
```

### 8.3. Render HTML form

#### 8.3.1. Load stylesheets and scripts in head

```html
<!doctype html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, user-scalable=no, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Pet Form</title>

    <script src="http://code.jquery.com/jquery-3.3.1.min.js" integrity="sha256-FgpCb/KJQlLNfOu91ta32o/NMZxltwRo8QtmkMRdAu8="
        crossorigin="anonymous"></script>
    <!-- Latest compiled and minified CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u"
        crossorigin="anonymous">

    <!-- Optional theme -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap-theme.min.css" integrity="sha384-rHyoN1iRsVXV4nD0JutlnGaslCJuC7uwjduW9SVrLvRYooPp2bWYgmgJQIXwl/Sp"
        crossorigin="anonymous">

    <!-- Latest compiled and minified JavaScript -->
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js" integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa"
        crossorigin="anonymous"></script>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
```

#### 8.3.2. Form in body

```html
    <body>
        <div class="container">
            <div class="jumbotron" style="text-align: center">
                <h1>Pet Pal</h1>
                <p>Enter your pet's details below!</p>
            </div>
            <div class="row">
                <div class="col-md-offset-2 col-md-8">
                    <form method="POST" action="/send" role="form", name="petform">
                        <div class="form-group">
                            <label for="inputName">Pet Name</label>
                            <input type="text" class="form-control" id="inputName" name="petName" placeholder="Name">
                        </div>
                        <div class="form-group">
                            <label for="inputLat">Pet Location Lat</label>
                            <input type="text" class="form-control" id="inputLat" name="petLat" placeholder="Lat">
                        </div>
                        <div class="form-group">
                            <label for="inputLon">Pet Location Lon</label>
                            <input type="text" class="form-control" id="inputLon" name="petLon" placeholder="Lon">
                        </div>
                        <input type="submit" value="submit" class="btn btn-default">
                    </form>
                </div>
            </div>

        </div>

    </body>

</html>
```

### 8.4. JS to build plot

```js
function buildPlot() {
    /* data route */
  var url = "/api/pals";
  d3.json(url).then(function(response) {

    console.log(response);

    var data = response;

    var layout = {
      scope: "usa",
      title: "Pet Pals",
      showlegend: false,
      height: 600,
            // width: 980,
      geo: {
        scope: "usa",
        projection: {
          type: "albers usa"
        },
        showland: true,
        landcolor: "rgb(217, 217, 217)",
        subunitwidth: 1,
        countrywidth: 1,
        subunitcolor: "rgb(255,255,255)",
        countrycolor: "rgb(255,255,255)"
      }
    };

    Plotly.newPlot("plot", data, layout);
  });
}

buildPlot();
```
