# Leaflet

[Leaflet documentation](https://leafletjs.com/)

<!-- TOC -->

- [Leaflet](#leaflet)
  - [1. Map basics](#1-map-basics)
    - [1.1. Base Map](#11-base-map)
    - [1.2. A default marker with popup](#12-a-default-marker-with-popup)
    - [1.3. Customized markers](#13-customized-markers)
    - [1.4. Circle markers with popup programatically](#14-circle-markers-with-popup-programatically)
    - [1.5. Layer control](#15-layer-control)
  - [2. Different map types](#2-different-map-types)
    - [2.1. GeoJSON Scatters](#21-geojson-scatters)
    - [2.2. GeoJSON cloropleths with event listeners](#22-geojson-cloropleths-with-event-listeners)
    - [2.3. Heatmap plugin](#23-heatmap-plugin)
    - [2.4. Marker cluster plugin](#24-marker-cluster-plugin)
    - [2.5. Choropleth plugin](#25-choropleth-plugin)

<!-- /TOC -->

## 1. Map basics

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Basic Map</title>

  <!-- Leaflet CSS & JS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.0.2/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.0.2/dist/leaflet.js"></script>

  <!-- D3 JS -->
  <script src="https://d3js.org/d3.v4.min.js"></script>
  <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.1.1/jquery.min.js"></script>

  <!-- Leaflet Heatmap JS -->
  <script type="text/javascript" src="leaflet-heat.js"></script>

  <!-- Marker Cluster JS & CSS -->
  <script type="text/javascript" src="https://unpkg.com/leaflet.markercluster@1.0.3/dist/leaflet.markercluster.js"></script>
  <link rel="stylesheet" type="text/css" href="https://unpkg.com/leaflet.markercluster@1.0.3/dist/MarkerCluster.css">
  <link rel="stylesheet" type="text/css" href="https://unpkg.com/leaflet.markercluster@1.0.3/dist/MarkerCluster.Default.css">

  <!-- Leaflet-Choropleth JavaScript -->
  <script type="text/javascript" src="choropleth.js"></script>

  <!-- My CSS -->
  <link rel="stylesheet" type="text/css" href="style.css">
</head>

<body>
  <!-- The div where we will inject our map -->
  <div id="map"></div>

  <!-- API key -->
  <script type="text/javascript" src="config.js"></script>
  <!-- My JS -->
  <script type="text/javascript" src="logic.js"></script>
</body>
</html>
```

### 1.1. Base Map

```js
// Creating our initial map object
// We set the longitude, latitude, and the starting zoom level
// This gets inserted into the div with an id of 'map'
var myMap = L.map("map", {
  center: [45.52, -122.67],
  zoom: 13
});

// Adding a tile layer (the background map image) to our map
// We use the addTo method to add objects to our map
L.tileLayer("https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}", {
  attribution: "Map data &copy; <a href=\"https://www.openstreetmap.org/\">OpenStreetMap</a> contributors, <a href=\"https://creativecommons.org/licenses/by-sa/2.0/\">CC-BY-SA</a>, Imagery © <a href=\"https://www.mapbox.com/\">Mapbox</a>",
  maxZoom: 18,
  id: "mapbox.streets",
  accessToken: API_KEY
}).addTo(myMap);
```

### 1.2. A default marker with popup

```js
// Create a new marker
// Pass in some initial options, and then add it to the map using the addTo method
var marker = L.marker([45.52, -122.67], {
  draggable: true,
  title: "My First Marker"
}).addTo(myMap);

// Binding a pop-up to our marker
marker.bindPopup("Hello There!"); // can also have html inside
```

`marker._latlng` to find dragged marker's coordinates

### 1.3. Customized markers

[documentation on path methods](https://leafletjs.com/reference-1.3.2.html#path)

- Initialize map object

  ```js
  // Create an initial map object
  // Set the longitude, latitude, and the starting zoom level
  var myMap = L.map("map").setView([45.52, -122.67], 13);

  var API_KEY = "pk.eyJ1IjoiYWxmN3RmIiwiYSI6ImNqa2l4b2RuczE5aXEzcW9hcGM0eHYxOWEifQ.qV8hIE76xSAmxTv1dkT2zA";

  // Add a tile layer (the background map image) to our map
  // Use the addTo method to add objects to our map
  L.tileLayer("https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}", {
    attribution: "Map data &copy; <a href=\"https://www.openstreetmap.org/\">OpenStreetMap</a> contributors, <a href=\"https://creativecommons.org/licenses/by-sa/2.0/\">CC-BY-SA</a>, Imagery © <a href=\"https://www.mapbox.com/\">Mapbox</a>",
    maxZoom: 18,
    id: "mapbox.streets",
    accessToken: API_KEY
  }).addTo(myMap);
  ```

- Create a new marker

  ```js
  L.marker([45.52, -122.67]).addTo(myMap);
  ```

- Create a circle and pass in some initial options

  ```js
  L.circle([45.52, -122.69], {
    color: "green",
    fillColor: "green",
    fillOpacity: 0.75,
    radius: 500 // in meters
  }).addTo(myMap);
  ```

- Create a Polygon and pass in some initial options

  ```js
  L.polygon([
    [45.54, -122.68],
    [45.55, -122.68],
    [45.55, -122.66]
  ], {
    color: "yellow",
    fillColor: "yellow",
    fillOpacity: 0.75
  }).addTo(myMap);
  ```

- Polyline

  ```js
  // Coordinates for each point to be used in the polyline
  var line = [
    [45.51, -122.68],
    [45.50, -122.60],
    [45.48, -122.70],
    [45.54, -122.75]
  ];

  // Create a polyline using the line coordinates and pass in some initial options
  L.polyline(line, {
    color: "red"
  }).addTo(myMap);
  ```

- Create a rectangle and pass in some initial options

  ```js
  L.rectangle([
    [45.55, -122.64],
    [45.54, -122.61]
  ], {
    color: "black",
    weight: 3,
    stroke: true
  }).addTo(myMap);
  ```

### 1.4. Circle markers with popup programatically

- Initialize map object

  ```js
  // Creating map object
  var myMap = L.map("map", {
    center: [37.09, -95.71],
    zoom: 5
  });

  // Adding tile layer
  var API_KEY = "pk.eyJ1IjoiYWxmN3RmIiwiYSI6ImNqa2l4b2RuczE5aXEzcW9hcGM0eHYxOWEifQ.qV8hIE76xSAmxTv1dkT2zA";

  // Add a tile layer (the background map image) to our map
  // Use the addTo method to add objects to our map
  L.tileLayer("https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}", {
    attribution: "Map data &copy; <a href=\"https://www.openstreetmap.org/\">OpenStreetMap</a> contributors, <a href=\"https://creativecommons.org/licenses/by-sa/2.0/\">CC-BY-SA</a>, Imagery © <a href=\"https://www.mapbox.com/\">Mapbox</a>",
    maxZoom: 18,
    id: "mapbox.streets",
    accessToken: API_KEY
  }).addTo(myMap);
  ```

- Data

  ```js
  // An array containing each city's name, location, and population
  var cities = [
    {
      location: [40.7128, -74.0059],
      name: "New York",
      population: 8550405
    },
    {
      location: [41.8781, -87.6298],
      name: "Chicago",
      population: 2720546
    },
  ];
  ```

- Loop through for markers and popups

  ```js
  // Loop through the cities array and create one marker for each city, bind a popup containing its name and population and add it to the map
  for (var i = 0; i < cities.length; i++) {
    var city = cities[i];
    L.circle(city.location, {
      color: "white",
      fillColor: "purple",
      fillOpacity: 0.75,
      radius: Math.sqrt(city.population) *30
    })
      .bindPopup("<h1>" + city.name + "</h1> <hr> <h3>Population " + city.population + "</h3>")
      .addTo(myMap);
  }
  ```

### 1.5. Layer control

- Data

  ```js
  // An array containing all of the information needed to create city and state markers
  var locations = [
    {
      coordinates: [40.7128, -74.0059],
      state: {
        name: "New York State",
        population: 19795791
      },
      city: {
        name: "New York",
        population: 8550405
      }
    },
    {
      coordinates: [34.0522, -118.2437],
      state: {
        name: "California",
        population: 39250017
      },
      city: {
        name: "Lost Angeles",
        population: 3971883
      }
    },
  ];
  ```

- Layer group of circle markers

  ```js
  // Define arrays to hold created city and state markers
  var cityMarkers = [];
  var stateMarkers = [];

  // Loop through locations and create city and state markers
  for (var i = 0; i < locations.length; i++) {
    // Set the marker radius for the state by passing population into the markerSize function
    stateMarkers.push(
      L.circle(locations[i].coordinates, {
        stroke: false,
        fillOpacity: 0.75,
        color: "white",
        fillColor: "white",
        radius: markerSize(locations[i].state.population)
      })
    );

    // Set the marker radius for the city by passing population into the markerSize function
    cityMarkers.push(
      L.circle(locations[i].coordinates, {
        stroke: false,
        fillOpacity: 0.75,
        color: "purple",
        fillColor: "purple",
        radius: markerSize(locations[i].city.population)
      })
    );
  }

  // Create marker layers
  var cities = L.layerGroup(cityMarkers);
  var states = L.layerGroup(stateMarkers);
  ```

  Layer group of default markers with popups

  ```js
  // An array which will be used to store created cityMarkers
  var cityMarkers = [];

  for (var i = 0; i < cities.length; i++) {
    // loop through the cities array, create a new marker, push it to the cityMarkers array
    cityMarkers.push(
      L.marker(cities[i].location).bindPopup("<h1>" + cities[i].name + "</h1>")
    );
  }

  // Add all the cityMarkers to a new layer group.
  // Now we can handle them as one group instead of referencing each individually
  var cityLayer = L.layerGroup(cityMarkers);
  ```

- Basemap tiles

  ```js
  // light map tiles
  var light = L.tileLayer("https://api.mapbox.com/styles/v1/mapbox/outdoors-v10/tiles/256/{z}/{x}/{y}?" +
    "access_token=pk.eyJ1IjoiYWxmN3RmIiwiYSI6ImNqa2l4b2RuczE5aXEzcW9hcGM0eHYxOWEifQ.qV8hIE76xSAmxTv1dkT2zA");

  // dark map tiles
  var dark = L.tileLayer("https://api.mapbox.com/styles/v1/mapbox/dark-v9/tiles/256/{z}/{x}/{y}?" +
    "access_token=pk.eyJ1IjoiYWxmN3RmIiwiYSI6ImNqa2l4b2RuczE5aXEzcW9hcGM0eHYxOWEifQ.qV8hIE76xSAmxTv1dkT2zA");
  ```

- Create basemaps

  ```js
  // Create a baseMaps object to contain the streetmap and darkmap
  var baseMaps = {
    Light: light,
    Dark: dark
  };
  ```

- Create overlay maps

  ```js
  // Create an overlayMaps object here to contain the "State Population" and "City Population" layers
  var overlayMaps = {
    "City Population": cities,
    "State Population": states
  };
  ```

- Set map default

  ```js
  // Modify the map so that it will have the streetmap, states, and cities layers
  var myMap = L.map("map", {
    center: [37.1, -95.7],
    zoom: 5,
    layers: [light, states, cities]
  });
  ```

- Create layer control

  ```js
  // Create a layer control, containing our baseMaps and overlayMaps, and add them to the map
  L.control.layers(baseMaps, overlayMaps, {collapsed: true}).addTo(myMap);
  ```

## 2. Different map types

### 2.1. GeoJSON Scatters

See Leaflet Documentation [documentation](http://leafletjs.com/reference.html#geojson) and [examples](http://leafletjs.com/examples/geojson/) on GeoJSON

- API query

  ```js
  // Store our API endpoint as queryUrl
  var queryUrl = "http://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2014-01-01&endtime=" +
    "2014-01-02&maxlongitude=-69.52148437&minlongitude=-123.83789062&maxlatitude=48.74894534&minlatitude=25.16517337";

  // Perform a GET request to the query URL
  d3.json(queryUrl, function(data) {
    console.log(data.features);
  ```

- GeoJSON layer

  ```js
    // Using the features array sent back in the API data, create a GeoJSON layer and add it to the map
    var earthquakes = L.geoJSON(data, {
      onEachFeature: (feature, layer) => {
        layer.bindPopup(feature.properties.place);
      }
    });
  ```

- Basemap tiles

  ```js
    // Define streetmap and darkmap layers
    var streetmap = L.tileLayer("https://api.mapbox.com/styles/v1/mapbox/outdoors-v10/tiles/256/{z}/{x}/{y}?" +
      "access_token=pk.eyJ1IjoiYWxmN3RmIiwiYSI6ImNqa2l4b2RuczE5aXEzcW9hcGM0eHYxOWEifQ.qV8hIE76xSAmxTv1dkT2zA");

    var darkmap = L.tileLayer("https://api.mapbox.com/styles/v1/mapbox/dark-v9/tiles/256/{z}/{x}/{y}?" +
      "access_token=pk.eyJ1IjoiYWxmN3RmIiwiYSI6ImNqa2l4b2RuczE5aXEzcW9hcGM0eHYxOWEifQ.qV8hIE76xSAmxTv1dkT2zA");
  ```

- Basemap and overlay map objects

  ```js
    // Define a baseMaps object to hold our base layers
    var baseMaps = {
      "Street Map": streetmap,
      "Dark Map": darkmap
    };

    var overlayMaps = {
      "Earthquakes": earthquakes,
    };
  ```

- Map with layer control

  ```js
    // Create a new map
    var myMap = L.map("map", {
      center: [
        37.09, -95.71
      ],
      zoom: 5,
      layers: [streetmap, earthquakes]
    });

    // Create a layer control containing our baseMaps
    // Be sure to add an overlay Layer containing the earthquake GeoJSON
    L.control.layers(baseMaps, overlayMaps, {collapsed: false}).addTo(myMap);

  });
  ```

  <img src="resources/scatters.png">

### 2.2. GeoJSON cloropleths with event listeners

- Basemap tiles

  ```js
  // Creating map object
  var map = L.map("map", {
    center: [40.7128, -74.0059],
    zoom: 11
  });

  // Adding tile layer
  L.tileLayer("https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}", {
    attribution: "Map data &copy; <a href=\"https://www.openstreetmap.org/\">OpenStreetMap</a> contributors, <a href=\"https://creativecommons.org/licenses/by-sa/2.0/\">CC-BY-SA</a>, Imagery © <a href=\"https://www.mapbox.com/\">Mapbox</a>",
    maxZoom: 18,
    id: "mapbox.streets",
    accessToken: API_KEY
  }).addTo(map);
  ```

- Define colors

  ```js
  // Function that will determine the color of a neighborhood based on the borough it belongs to
  function chooseColor(borough) {
    switch (borough) {
    case "Brooklyn":
      return "yellow";
    case "Bronx":
      return "red";
    case "Manhattan":
      return "orange";
    case "Queens":
      return "green";
    case "Staten Island":
      return "purple";
    default:
      return "black";
    }
  }
  ```

- API URL

  ```js
  var link = "http://data.beta.nyc//dataset/0ff93d2d-90ba-457c-9f7e-39e47bf2ac5f/resource/" +
  "35dd04fb-81b3-479b-a074-a27a37888ce7/download/d085e2f8d0b54d4590b1e7d1f35594c1pediacitiesnycneighborhoods.geojson";
  ```

- Grab and plot GeoJSON data

  ```js
  d3.json(link, function(data) {
    
    // Creating a geoJSON layer with the retrieved data
    L.geoJson(data, {

      // Style each feature (in this case a neighborhood)
      style: function(feature) {
        return {
          color: "white",
          // Call the chooseColor function to decide which color to color our neighborhood (color based on borough)
          fillColor: chooseColor(feature.properties.borough),
          fillOpacity: 0.5,
          weight: 1.5
        };
      },

      // Event listeners called on each feature
      onEachFeature: function(feature, layer) {

        // Set mouse events to change map styling
        layer.on({

          // When a user's mouse touches a map feature, the mouseover event calls this function, that feature's opacity changes to 90% so that it stands out
          mouseover: function(event) {
            layer = event.target;
            layer.setStyle({
              fillOpacity: 0.9
            });
          },

          // When the cursor no longer hovers over a map feature - when the mouseout event occurs - the feature's opacity reverts back to 50%
          mouseout: function(event) {
            layer = event.target;
            layer.setStyle({
              fillOpacity: 0.5
            });
          },

          // When a feature (neighborhood) is clicked, it is enlarged to fit the screen
          click: function(event) {
            map.fitBounds(event.target.getBounds());
          }
        });

        // Giving each feature a pop-up with information pertinent to it
        layer.bindPopup("<h1>" + feature.properties.neighborhood + "</h1> <hr> <h2>" + feature.properties.borough + "</h2>");

      }
    }).addTo(map);
  });
  ```

  <img src="resources/choropleths.png">

### 2.3. Heatmap plugin

Documentation on [leaflet plugins](https://leafletjs.com/plugins.html)

- Basemap tiles

  ```js
  var myMap = L.map('map', {
    center: [37.7749, -122.4194],
    zoom: 13
  });

  L.tileLayer('https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}', {
    attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, <a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery © <a href="https://www.mapbox.com/">Mapbox</a>',
    maxZoom: 18,
    id: 'mapbox.streets',
    accessToken: API_KEY
  }).addTo(myMap);
  ```

- Plot heatmap using data from URL

  ```js
  var url = "https://data.sfgov.org/resource/gxxq-x39z.json?$limit=10000"

  d3.json(url, function(response){

    console.log(response);

    var heatArray = [];

    for (var i = 0; i < response.length; i++) {
      var location = response[i].location;

      if (location) {
        heatArray.push([location.latitude, location.longitude])
      }
    }

    var heat = L.heatLayer(heatArray, {
      radius: 20,
      blur: 35
    }).addTo(myMap)

  });
  ```

  <img src="resources/heatmap.png">


### 2.4. Marker cluster plugin

Documentation on [leaflet marker clustering](https://github.com/Leaflet/Leaflet.markercluster)

- Basemap tiles

  ```js
  // Creating map object
  var myMap = L.map("map", {
    center: [40.7, -73.95],
    zoom: 11
  });

  // Adding tile layer to the map
  L.tileLayer("https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}", {
    attribution: "Map data &copy; <a href=\"https://www.openstreetmap.org/\">OpenStreetMap</a> contributors, <a href=\"https://creativecommons.org/licenses/by-sa/2.0/\">CC-BY-SA</a>, Imagery © <a href=\"https://www.mapbox.com/\">Mapbox</a>",
    maxZoom: 18,
    id: "mapbox.streets",
    accessToken: API_KEY
  }).addTo(myMap);
  ```

- API URL

  ```js
  // Building API query URL
  var baseURL = "https://data.cityofnewyork.us/resource/fhrw-4uyv.json?";
  var date = "$where=created_date between'2016-01-10T12:00:00' and '2017-01-01T14:00:00'";
  var complaint = "&complaint_type=Rodent";
  var limit = "&$limit=10000";

  // Assembling API query URL
  var url = baseURL + date + complaint + limit;
  ```

- Grab and plot data

  ```js
  // Grabbing the data with d3..
  d3.json(url, function(response) {

    // Creating a new marker cluster group
    var markers = L.markerClusterGroup();

    // Loop through our data...
    for (var i = 0; i < response.length; i++) {
      // set the data location property to a variable
      var location = response[i].location;

      // If the data has a location property...
      if (location) {

        // Add a new marker to the cluster group and bind a pop-up
        markers.addLayer(L.marker([location.coordinates[1], location.coordinates[0]])
          .bindPopup(response[i].descriptor));
      }

    }

    // Add our marker cluster layer to the map
    myMap.addLayer(markers);

  });
  ```

  <img src="resources/marker_clusters.png">

### 2.5. Choropleth plugin

Documentation on [leaflet choropleth plugin](https://github.com/timwis/Leaflet-choropleth)

- Basemap tiles

  ```js
  // Creating map object
  var myMap = L.map("map", {
    center: [40.7128, -74.0059],
    zoom: 11
  });

  // Adding tile layer
  L.tileLayer("https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}", {
    attribution: "Map data &copy; <a href=\"https://www.openstreetmap.org/\">OpenStreetMap</a> contributors, <a href=\"https://creativecommons.org/licenses/by-sa/2.0/\">CC-BY-SA</a>, Imagery © <a href=\"https://www.mapbox.com/\">Mapbox</a>",
    maxZoom: 18,
    id: "mapbox.streets",
    accessToken: API_KEY
  }).addTo(myMap);
  ```

- API URL

  ```js
  // Link to GeoJSON
  var APILink = "http://data.beta.nyc//dataset/d6ffa9a4-c598-4b18-8caf-14abde6a5755/resource/74cdcc33-512f-439c-" +
  "a43e-c09588c4b391/download/60dbe69bcd3640d5bedde86d69ba7666geojsonmedianhouseholdincomecensustract.geojson";

  var geojson;
  ```

- Grab and plot data

  The [colorbrewer2](http://colorbrewer2.org/) website provides color schemes (in hex values) that you can use to customize a choropleth map.

  ```js
  // Grabbing data with d3...
  d3.json(APILink, function(data) {

    // Creating a new choropleth layer
    geojson = L.choropleth(data, {
      // Which property in the features to use
      valueProperty: "MHI",
      // Color scale
      scale: ["#ffffb2", "#b10026"],
      // Number of breaks in step range
      steps: 10,
      // q for quartile, e for equidistant, k for k-means
      mode: "q",
      style: {
        // Border color
        color: "#fff",
        weight: 1,
        fillOpacity: 0.8
      },
      // Binding a pop-up to each layer
      onEachFeature: function(feature, layer) {
        layer.bindPopup(feature.properties.COUNTY + " " + feature.properties.State + "<br>Median Household Income:<br>" +
          "$" + feature.properties.MHI);
      }
    }).addTo(myMap);
  ```

- Add a legend

  Examples and [Leaflet documentation](https://github.com/timwis/leaflet-choropleth/blob/gh-pages/examples/legend/) on how to add a legend.

  ```js
    // Setting up the legend
    var legend = L.control({ position: "bottomright" });
    legend.onAdd = function() {
      var div = L.DomUtil.create("div", "info legend");
      var limits = geojson.options.limits;
      var colors = geojson.options.colors;
      var labels = [];

      // Add min & max
      var legendInfo = "<h1>Median Income</h1>" +
        "<div class=\"labels\">" +
          "<div class=\"min\">" + limits[0] + "</div>" +
          "<div class=\"max\">" + limits[limits.length - 1] + "</div>" +
        "</div>";

      div.innerHTML = legendInfo;

      limits.forEach(function(limit, index) {
        labels.push("<li style=\"background-color: " + colors[index] + "\"></li>");
      });

      div.innerHTML += "<ul>" + labels.join("") + "</ul>";
      return div;
    };

    // Adding legend to the map
    legend.addTo(myMap);

  });
  ```

  <img src="resources/choropleths2.png">
