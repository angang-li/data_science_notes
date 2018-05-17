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

#### Database principles
* atomicity: a transaction happens as a whole or not at all

<br>

<!-- TOC -->

- [**Relational Databases**](#relational-databases)
- [SQL elements](#sql-elements)
    - [1. Select clauses](#1-select-clauses)
    - [2. Insert statement](#2-insert-statement)
    - [3. Update statement](#3-update-statement)
    - [4. Create statement](#4-create-statement)
    - [5. Delete statement](#5-delete-statement)
- [Python DB-API](#python-db-api)
    - [4. Write code with DB-API](#4-write-code-with-db-api)
    - [5. PostgreSQL command line](#5-postgresql-command-line)

<!-- /TOC -->


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
    <br>

## 3. Update statement
* **3.1. update away the spam**
    |||
    |---|---|
    |`update animals` <br>&nbsp;&nbsp;&nbsp;&nbsp; `set name = 'cheese'` <br>&nbsp;&nbsp;&nbsp;&nbsp; `where name like '%awful%';`||
    <br>

## 4. Create statement
* **4.1. create a table**
    |||
    |---|---|
    |`create table animals (` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `name text,` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `species text,` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `birthdate date);`||
    <br>

## 5. Delete statement
* **5.1. delete spam**
    |||
    |---|---|
    |`delete from animals` <br>&nbsp;&nbsp;&nbsp;&nbsp; `where name = 'cheese';`||

<br>

# Python DB-API
Python DB-API is a standard for python libraries that let the code to connect to the datebases. <br>

## 4. Write code with DB-API
* **4.1. basic structure** <br>
    .connect(...) <br> &nbsp; &nbsp; &nbsp; ↓ <br>
    **connection**.commit() <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; .rollback() <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; .cursor() <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ↓ <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;**cursor**.execute(query) <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;.fetchone() <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;.fetchall()

* **4.2. select with SQLite**
    ```python
    import sqlite3
    conn = sqlite3.connect('Cookies')
    cursor = conn.cursor()
    cursor.execute(
        'select host_key from cookies limit 10')
    results = cursor.fetchall()
    print(results)
    conn.close()
    ```

* **4.3. select with PostgreSQL**
    ```python
    import psycopg2
    psycopg2.connect("dbname=bears")
    ```

* **4.4. insert with SQLite**
    ```python
    import sqlite3
    conn = sqlite3.connect('Cookies')
    cursor = conn.cursor()
    cursor.execute(
        "insert into names values ('Jennifer Smith')")
    conn.commit()
    conn.close()
    ```

* **4.5. solve SQL injection attack** <br>
    Use query parameters instead of string substitution to execute insert query
    ```python
    cursor.execute(
        "insert into names values (%s)", (content, ))
    ```

* **4.6. solve script injection attack**
    * `bleach.clean(...)` see [documentation](http://bleach.readthedocs.io/en/latest/)
    * `update` command

<br>

## 5. PostgreSQL command line
[PostgreSQL documentation](https://www.postgresql.org/docs/9.4/static/app-psql.html)
* **5.1. connect to a database named `forum`** <br>
    `psql forum`

* **5.2. get info about the database** <br>
    * display columns of a table named `posts` <br>
        `\d posts`

    * list all the tables in the database <br>
        `\dt`

    * list tables plus additional information <br>
        `\dt+`

    * switch between printing tables in plain text vs HTML
        `\H`

* **5.3. display the contents of a table named `posts` and refresh it every two seconds** <br>
    `select * from posts \watch`

