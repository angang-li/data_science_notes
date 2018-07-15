# Flask

[Flask](http://flask.pocoo.org/):
a micro web framework written in Python

* Use Flask to create and run a server
* Define endpoints using Flask's @app.route decorator
* Execute database queries on behalf of the client
* Return JSONified query results from API endpoints

[Django](https://www.djangoproject.com/): 
a free and open-source web framework, written in Python

<!-- TOC -->

- [Flask](#flask)
    - [1. API endpoint for jasonified data](#1-api-endpoint-for-jasonified-data)
    - [2. Database backed API](#2-database-backed-api)

<!-- /TOC -->

## 1. API endpoint for jasonified data

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