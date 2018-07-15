# NoSQL Database with MongoDB

<!-- TOC -->

- [NoSQL Database with MongoDB](#nosql-database-with-mongodb)
    - [1. Intro to MongoDB](#1-intro-to-mongodb)
        - [1.1. NoSQL database](#11-nosql-database)
        - [1.2. MongoDB](#12-mongodb)
    - [2. Mongo queries](#2-mongo-queries)
        - [2.1. Create a database](#21-create-a-database)
        - [2.2. Create collection and insert documents](#22-create-collection-and-insert-documents)
        - [2.3. Read documents in the collection](#23-read-documents-in-the-collection)
        - [2.4. Update documents](#24-update-documents)
        - [2.3. Delete document, collection, or database](#23-delete-document-collection-or-database)

<!-- /TOC -->

## 1. Intro to MongoDB

### 1.1. NoSQL database

- NoSQL database is often called "non SQL", "non relational", or "not only SQL" database.
- Comparison between NoSQL vs. SQL databases

    - (+) Flexible, can add new columns or fields on MongoDB without affecting existing rows or application performance
    - (+) Scalable, allowing almost unlimited growth
    - (-) Lack the ability to perform ACID transactions that span multiple pieces of data within the database

    | SQL | noSQL |
    | --- | --- |
    | relies on joins to combine relevant data | effectively JSONs |
    | table | collection |
    | row | document |
    | column | field |
    | vertically scalable, can increase the load on a single server by increasing things like CPU, RAM or SSD | horizontally scalable, can handle more traffic by sharding, or adding more servers in your NoSQL database |
    | good for any business that will benefit from its pre-defined structure and set schemas | good for businesses that have rapid growth or databases with no clear schema definitions |

### 1.2. MongoDB

- MongoDB is a type of NoSQL database.
- MongoDB uses a document-oriented model as opposed to a table-based relational model (SQL).
- MongoDB stores data in BSON format (effectively compressed JSON)

## 2. Mongo queries

### 2.1. Create a database

- Show existing database names

    `show dbs`

- Create and use database

    `use travel_db`

- Show the current database

    `db`

- Show existing collections

    `show collections`

### 2.2. Create collection and insert documents

- Insert one document, create collection if not exist

    `db.countries.insert({country:"Morocco", continent:"Africa", cities: ["Fez", "Casablanca", "Marrakech"]})`

- Insert many documents

    `db.countries.insertMany([{country: "Poland", continent: "Europe", cities: ["Krakow", "Warsaw", "Wroclaw"]},{country: "Nigeria", continent: "Africa", cities: ["Lagos"]}])`

### 2.3. Read documents in the collection

- Query all

    `db.countries.find().pretty()`

- Query by one attribute

    `db.countries.find({continent:"Africa"}).pretty()`

- Query with logic and

    `db.countries.find({cities: "Fez", continent: "Africa"})`

- Query with logic or

    `db.countries.find({$or: [{cities: "Fez"}, {continent: "Africa"}]})`

- Query by document id

    `db.countries.find("5b43f9cac55a4d03d8352c83")` <br>
    `db.countries.find({_id: ObjectId("5b43f9cac55a4d03d8352c83")})`

- Query with aggregation

    `db.countries.find().count()`

### 2.4. Update documents

- Update the first matching entry

    `db.countries.update({country:"Poland"}, {$set: {continent: "Europe"}})`

- Update multiple matching entries

    `db.countries.update({"continent": "Africa"}, {$set: {"continent": "Antarctica"}}, {multi: true})` <br>
    alternatively, `db.countries.updateMany({"continent": "Africa"}, {$set: {"continent": "Antarctica"}})`

- Insert new values into matching entry

    `db.countries.update({country:"Poland"}, {$push: {cities: "Gdansk"}})`

- Update and insert

    Update if one exists, otherwise insert new <br>
    `db.countries.update({country:"Austria"}, {$set: {continent: "Europe"}}, {upsert: true})`

- Add value by 1 (increment)

    `db.divers.updateMany({}, {$inc: {yearsDiving: 1}})`

### 2.3. Delete document, collection, or database

- Remove matching documents

    `db.countries.remove({continent: "Asia"})`

- Empty a collection

    `db.countries.remove({})`

- Drop collection

    `db.countries.drop()`

- Drop database

    `db.dropDatabase()`
