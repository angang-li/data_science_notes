# **Relational Databases**

#### Databases:
* persistent storage
* safe concurrent access by multiple programs and users

#### Relational databases:
* flexible query languages with aggregation and join operations
* constraining rules for protecting consistency of data

#### Data types:
* text, char(n), varchar(n)
* integer, real, double precision, decimal
* date, time, timestamp<br>
<br>
Always write 'single quotes' around text strings and date/time values

<br>

# SQL elements
## 1. Select clauses
* **1.1. Where, limit, offset**

    **select** *columns* **from** *tables* **where** *condition* **;**
    * Columns are separated by commas; use * to select all columns.
    * The condition is Boolean (`and`, `or`, `not`). [DeMorgan's Law](https://en.wikipedia.org/wiki/De_Morgan%27s_laws) 
    * Comparison operators (`=`, `!=`, `<`, `>=`, etc.) <br>

    |||
    |---|---|
    |`select food` <br> `from diet` <br> `where species = 'llama'` <br> `and name = 'Max` <br> `limit 10` <br> `offset 20;`|column name <br> table name <br> row restriction <br>  <br>10 rows<br>skip 20 rows<br>|

* **1.2. Order by**
    |||
    |---|---|
    |`select *` <br> `from animals` <br> `where species = 'orangutan'` <br> `order by birthdate desc;`| <br><br><br>ascending if not specified<br>|

* **1.3. Group by with aggregations**
    |||
    |---|---|
    |`select species, min(birthdate)` <br> `from animals` <br> `group by species;`||

    |||
    |---|---|
    |`select name, count(*) as num` <br> `from animals` <br> `group by name` <br> `order by num desc` <br> `limit 5;` <br>||

* **1.4. Having**

    * `Where` is a restriction on the source tables
    * `Having` is a restriction on the results, after aggregation <br>

    |||
    |---|---|
    |`select species, count(*) as num` <br> `from animals` <br> `group by species` <br> `having num = 1;` <br>||

* **1.5. Join** <br>
    Primary key must be unique <br>
    * Generally <br>

    |||
    |---|---|
    |`select animals.name, animals.species, diet.food` <br> `from animals join diet` <br> `on animals.species = diet.species` <br> `where diet.food = 'fish';`| <br> join name, species, food <br> based on species <br> that eat fish|
    <br>

    * Simple join <br>

    |||
    |---|---|
    |`select animals.name, animals.species, diet.food` <br> `from animals, diet` <br> `where animals.species = diet.species;` <br> `and diet.food = 'fish';` | <br> join name, species, food <br> based on species <br> that eat fish|
    <br>

## 2. Insert statement
* **2.1. Generally**
    |||
    |---|---|
    |`insert into animals ( name, species, ... )` <br> `values ( 'Amy', 'opossum', ... );`||

* **2.2. If values have same order as columns and start from the first column**
    |||
    |---|---|
    |`insert into animals` <br> `values ( 'Amy', 'opossum', ... );`||

## 3. Create table
* 
    |||
    |---|---|
    |`create table animals (` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `name text,` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `species text,` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `birthdate date);`||