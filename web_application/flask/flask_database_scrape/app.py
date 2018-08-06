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