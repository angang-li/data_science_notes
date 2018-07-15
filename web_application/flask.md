# Flask

[Flask](http://flask.pocoo.org/):
a micro web framework written in Python <br>
[Django](https://www.djangoproject.com/): 
a free and open-source web framework, written in Python

<!-- TOC -->

- [Flask](#flask)
    - [1. API endpoint for jasonified data](#1-api-endpoint-for-jasonified-data)
    - [2. Database backed API](#2-database-backed-api)
    - [3. Render templates with flask](#3-render-templates-with-flask)
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

## 3. Render templates with flask

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

- [data.py](flask_database/data.py)
- [app.py](flask_database/app.py)
- [templates/index.html](flask_database/templates/index.html)

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

- [scraping.py](flask_database_scrape/scraping.py)
- [app.py](flask_database_scrape/app.py)
- [templates/index.html](flask_database_scrape/templates/index.html)

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