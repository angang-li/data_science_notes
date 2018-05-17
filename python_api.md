# API (Application Programming Interface)

```python
import requests
import json
from pprint import pprint
```

- **Send requests, retrieve API data**

```python
requests.get(url)
response = requests.get(url)
response.status_code
response.url # confirm URL
response_json = response.json()
json.dumps(response_json, indent=4, sort_keys=True)
```

- **Compile parameters in API query**

```python
params = {
    'city': 'london',
    'appid': api_key,
}
requests.get(url, params = params)
```

- **API wrappers**

[`openweathermapy` documentation](http://openweathermapy.readthedocs.io/en/latest/)

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

[`citypy` documentation](https://github.com/wingchen/citipy)

```python
from citipy import citipy
city = citipy.nearest_city(22.99, 120.21) # lat, lon
city.city_name
city.country_code
```
