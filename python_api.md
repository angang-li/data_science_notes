# API (Application Programming Interface)

<!-- TOC -->

- [API (Application Programming Interface)](#api-application-programming-interface)
    - [1. API request](#1-api-request)
        - [- ### 1.1. Request with URL](#11-request-with-url)
        - [- ### 1.2. Request with URL and params](#12-request-with-url-and-params)
        - [- ### 1.3. Google geocoding, places, and radar APIs](#13-google-geocoding--places--and-radar-apis)
    - [2. API wrappers](#2-api-wrappers)
        - [- ### 2.1. `openweathermapy`](#21-openweathermapyhttp---openweathermapyreadthedocsio-en-latest)
        - [- ### 2.2. `citypy`](#22-citypyhttps---githubcom-wingchen-citipy)
        - [- ### 2.3. `Census`](#23-censushttps---githubcom-commercedataservice-census-wrapper)

<!-- /TOC -->

## 1. API request

- ### 1.1. Request with URL

    ```python
    import requests
    import json
    from pprint import pprint

    response = requests.get(url)
    response.status_code # 200 is good
    response.url # confirm URL
    response_json = response.json()
    json.dumps(response_json, indent=4, sort_keys=True)
    ```

- ### 1.2. Request with URL and params

    [Openweathermap API]((https://openweathermap.org/api))

    ```python
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': 'london',
        'appid': api_key,
    }
    requests.get(base_url, params = params)
    ```

- ### 1.3. Google geocoding, places, and radar APIs

    [Geocoding](https://developers.google.com/maps/documentation/geocoding/intro): Get lat/lon based on address

    ```python
    import requests
    import json
    from config import gkey

    target_city = "Seattle, Washington"
    params = {"address": target_city, "key": gkey}
    target_url = "https://maps.googleapis.com/maps/api/geocode/json"

    response = requests.get(target_url, params=params).json()
    lat = response["results"][0]["geometry"]["location"]["lat"]
    lng = response["results"][0]["geometry"]["location"]["lng"]
    ```

    [Places](https://developers.google.com/maps/documentation/javascript/places#place_search_requests): Map search based on lat/lon

    ```python
    import requests
    import json
    from config import gkey
    
    target_coordinates = "47.6062095, -122.3320708"
    target_search = "bike"
    target_radius = 8000
    target_type = "store"

    # set up a parameters dictionary
    params = {
        "location": target_coordinates,
        "keyword": target_search,
        "radius": target_radius,
        "type": target_type,
        "key": gkey
    }

    # base url
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    # run a request using our params dictionary
    response = requests.get(base_url, params=params).json()
    # Print the name and address of the first restaurant that appears
    print(response["results"][0]["name"])
    print(response["results"][0]["vicinity"])
    ```

    Radar search: Count nearby map searches

    ```python
    import requests
    import json
    from config import gkey
    
    target_city = {"lat": 43.6187102, "lng": -116.2146068}
    target_coords = f"{target_city['lat']},{target_city['lng']}"

    params = {
        "location": target_coords,
        "radius": 8000,
        "keyword": "ice cream",
        "type": "food",
        "key": gkey
    }

    # Build the endpoint URL (Checks all ice cream shops)
    base_url = "https://maps.googleapis.com/maps/api/place/radarsearch/json"

    # Run a request to endpoint and convert result to json
    ice_cream_data = requests.get(base_url, params=params).json()
    print(len(ice_cream_data["results"]))
    ```

    Loop to get lat/lon and keyword search

    ```python
    # Dependencies
    import pandas as pd
    import numpy as np
    import requests
    import json
    # Import API key
    from config import gkey

    # Load data
    cities_pd = pd.read_csv("../Resources/Cities.csv")
    cities_pd.head()
    cities_pd['lat'] = ""
    cities_pd['lon'] = ""
    cities_pd['airport name'] = ""
    cities_pd['airport address'] = ""
    cities_pd['airport rating'] = ""
    ```
    ```python
    # Latitude and longitude
    target_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"key": gkey}

    for index, row in cities_pd.iterrows():
        target_city = row['City']
        params['address'] = target_city
        print(f"Retrieving Results for Index {index}: {target_city}.")
        
        response = requests.get(target_url, params=params).json()
        results = response.get('results')
        
        if (results):
            location = response['results'][0]["geometry"]["location"]
            print(f"Geo-location of {target_city} is {location.get('lat', 'None')}, {location.get('lng', 'None')}.")
            cities_pd.loc[index, 'lat'] = location.get('lat', '') # protect from keyError
            cities_pd.loc[index, 'lon'] = location.get('lng', '')
        else:
            print("No results for " + target_city)
        print("------------")
    ```
    ```python
    # Search based on lat lon
    target_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "rankby": "distance",
        "type": "airport",
        "key": gkey,
        "keyword": "international+airport",
    }
    
    for index, row in cities_pd.iterrows(): # use iterrows to iterate through pandas dataframe
        target_coordinates = f"{row['lat']}, {row['lon']}"
        target_city = cities_pd.loc[index, 'City']
        params['location'] = target_coordinates
        print(f"Retrieving Results for Index {index}: {target_city}.")
        
        response = requests.get(target_url, params=params).json()
        results = response.get('results')
        
        if (results):
            airp = response['results'][0]
            print(f"International airport of {target_city} is {airp.get('name', 'None')}.")
            cities_pd.loc[index, 'airport name'] = airp.get('name', '') # protect from keyError
            cities_pd.loc[index, 'airport address'] = airp.get('vicinity', '')
            cities_pd.loc[index, 'airport rating'] = airp.get('rating', '')
        else:
            print("No results for " + target_city)
        print("------------")
    ```

## 2. API wrappers

API wrapper acts as a shield from internal changes of json

- ### 2.1. [`openweathermapy`](http://openweathermapy.readthedocs.io/en/latest/)

    ```python
    import openweathermapy.core as owm

    # Create settings dictionary with information we're interested in
    settings = {"units": "metric", "appid": api_key}

    # Get current weather
    current_weather_paris = owm.get_current("Paris", **settings)

    # Get parameters of interest
    summary = ["name", "main.temp"]
    data = current_weather_paris(*summary)
    ```

- ### 2.2. [`citypy`](https://github.com/wingchen/citipy)

    ```python
    from citipy import citipy
    city = citipy.nearest_city(22.99, 120.21) # lat, lon
    city.city_name
    city.country_code
    ```

- ### 2.3. [`Census`](https://github.com/CommerceDataService/census-wrapper)

    [Parameter labels](https://gist.github.com/afhaque/60558290d6efd892351c4b64e5c01e9b)

    ```python
    # Run Census Search to retrieve data on all zip codes (2013 ACS5 Census)
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import requests
    from census import Census
    from config import api_key # Census API Key

    c = Census(api_key, year=2013)
    census_data = c.acs5.get(("NAME", "B19013_001E", "B01003_001E", "B01002_001E",
                            "B19301_001E", "B17001_002E"), 
                            {'for': 'zip code tabulation area:*'})

    # Convert to DataFrame
    census_pd = pd.DataFrame(census_data)

    # Column Reordering
    census_pd = census_pd.rename(columns={"B01003_001E": "Population",
                                        "B01002_001E": "Median Age",
                                        "B19013_001E": "Household Income",
                                        "B19301_001E": "Per Capita Income",
                                        "B17001_002E": "Poverty Count",
                                        "NAME": "Name", "zip code tabulation area": "Zipcode"})
    ```

