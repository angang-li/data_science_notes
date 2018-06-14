# **Relational Databases**

**Databases:**

* persistent storage
* safe concurrent access by multiple programs and users

**Relational databases:**

* flexible query languages with aggregation and join operations
* constraining rules for protecting consistency of data

**Data types:**

* text, char(n), varchar(n)
* integer, real, double precision, decimal
* date, time, timestamp

Always write 'single quotes' around text strings and date/time values
<br>

**Database principles:**

* atomicity: a transaction happens as a whole or not at all

<br>

<!-- TOC -->

- [**Relational Databases**](#relational-databases)
    - [1. Select clauses](#1-select-clauses)
    - [2. Insert statement](#2-insert-statement)
    - [3. Update statement](#3-update-statement)
    - [4. Create statement](#4-create-statement)
    - [5. Delete statement](#5-delete-statement)
    - [6. Write code with DB-API](#6-write-code-with-db-api)
    - [7. PostgreSQL command line](#7-postgresql-command-line)
    - [8. Deeper into SQL](#8-deeper-into-sql)

<!-- /TOC -->


## 1. Select clauses

* ### 1.1. Where, limit, offset

    **select** *columns* **from** *tables* **where** *condition* **;**
    * Columns are separated by commas; use * to select all columns.
    * The condition is Boolean (`and`, `or`, `not`). [DeMorgan's Law](https://en.wikipedia.org/wiki/De_Morgan%27s_laws) 
    * Comparison operators (`=`, `!=`, `<`, `>=`, etc.) <br>

    |||
    |---|---|
    |`select food` <br> `from diet` <br> `where species = 'llama'` <br> `and name = 'Max` <br> `limit 10` <br> `offset 20;`|column name <br> table name <br> row restriction <br>  <br>10 rows<br>skip 20 rows<br>|

* ### 1.2. Order by
    |||
    |---|---|
    |`select *` <br> `from animals` <br> `where species = 'orangutan'` <br> `order by birthdate desc;`| <br><br><br>ascending if not specified<br>|

* ### 1.3. Group by with aggregations
    |||
    |---|---|
    |`select species, min(birthdate)` <br> `from animals` <br> `group by species;`||

    |||
    |---|---|
    |`select name, count(*) as num` <br> `from animals` <br> `group by name` <br> `order by num desc` <br> `limit 5;` <br>||

* ### 1.4. Having

    * `Where` is a restriction on the source tables
    * `Having` is a restriction on the results, after aggregation <br>

    |||
    |---|---|
    |`select species, count(*) as num` <br> `from animals` <br> `group by species` <br> `having num = 1;` <br>||

* ### 1.5. Join
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

    * Left join to count what isn't there

    ```sql
    select products.name, products.sku, count(sales.sku) as num
      from products left join sales
        on products.sku = sales.sku
      group by products.sku;
    ```

    * Self Joins
    ```sql
    select a.id, b.id, a.building, a.room
            from residences as a, residences as b
      where a.building = b.building
        and a.room = b.room
        and a.id < b.id
      order by a.building, a.room;
    ```

## 2. Insert statement

* ### 2.1. Generally
    |||
    |---|---|
    |`insert into animals ( name, species, ... )` <br> `values ( 'Amy', 'opossum', ... );`||

* ### 2.2. If values have same order as columns and start from the first column
    |||
    |---|---|
    |`insert into animals` <br> `values ( 'Amy', 'opossum', ... );`||
    <br>

## 3. Update statement

* ### 3.1. update away the spam
    |||
    |---|---|
    |`update animals` <br>&nbsp;&nbsp;&nbsp;&nbsp; `set name = 'cheese'` <br>&nbsp;&nbsp;&nbsp;&nbsp; `where name like '%awful%';`||
    <br>

## 4. Create statement

* ### 4.1. create a table

    [PostgreSQL create table documentation](https://www.postgresql.org/docs/9.4/static/sql-createtable.html)

    Basic structure

    |||
    |---|---|
    |`create table animals (` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `name text [constraints],` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `species text [constraints],` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `birthdate date [constraints],` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `[row constraints]);`||
    <br>

    Assign primary key

    |||
    |---|---|
    |`create table animals (` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `id serial primary key,` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `name text,` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `birthdate date);`||
    <br>

    Assign multiple columns as primary key

    |||
    |---|---|
    |`create table animals (` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `postal_code text,` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `country text,` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `name text,` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `primary key (postal_code, country));`||
    <br>

    Declare relationships <br>
    Reference provides referential integrity - columns that are supposed to refer to each other are guaranteed to do so.
    
    |||
    |---|---|
    |`create table sale (` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `sku text reference products (sku),` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `sale_date date,` <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `count integer);`||
    <br>

## 5. Delete statement

* ### 5.1. delete spam
    |||
    |---|---|
    |`delete from animals` <br>&nbsp;&nbsp;&nbsp;&nbsp; `where name = 'cheese';`||

<br>

## 6. Write code with DB-API

Python DB-API is a standard for python libraries that let the code to connect to the datebases.

<br>

* ### 6.1. basic structure
    .connect(...) <br> &nbsp; &nbsp; &nbsp; ↓ <br>
    **connection**.commit() <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; .rollback() <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; .cursor() <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ↓ <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;**cursor**.execute(query) <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;.fetchone() <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;.fetchall()

* ### 6.2. select with SQLite
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

* ### 6.3. select with PostgreSQL
    ```python
    import psycopg2
    psycopg2.connect("dbname=bears")
    ```

* ### 6.4. insert with SQLite
    ```python
    import sqlite3
    conn = sqlite3.connect('Cookies')
    cursor = conn.cursor()
    cursor.execute(
        "insert into names values ('Jennifer Smith')")
    conn.commit()
    conn.close()
    ```

* ### 6.5. solve SQL injection attack
    Use query parameters instead of string substitution to execute insert query
    ```python
    cursor.execute(
        "insert into names values (%s)", (content, ))
    ```

* ### 6.6. solve script injection attack
    * `bleach.clean(...)` see [documentation](http://bleach.readthedocs.io/en/latest/)
    * `update` command

<br>

## 7. PostgreSQL command line

[PostgreSQL documentation](https://www.postgresql.org/docs/9.4/static/app-psql.html)

* ### 7.1. connect to a database named `forum`
    `psql forum` <br>
    if already within psql, `\c forum`

* ### 7.2. get info about the database
    * display columns of a table named `posts` <br>
        `\d posts`

    * list all the tables in the database <br>
        `\dt`

    * list tables plus additional information <br>
        `\dt+`

    * switch between printing tables in plain text vs HTML
        `\H`

* ### 7.3. display the contents of a table named `posts` and refresh it every two seconds
    `select * from posts \watch`

<br>

## 8. Deeper into SQL

* ### 8.1. Normalized vs denormalized table

    Rules for normalized tables:
  * Every row has the same number of columns.
  * There is a unique key, and everything in a row says something about the key.
  * Facts that don't relate to the key belong in different tables.
  * Tables shouldn't imply relationships that don't exist.

* ### 8.2. Creating and dropping a database or a table
  * `create database db_name [options];`
  * `drop database db_name [options];`
  * `drop table tb_name [options];`

* ### 8.3. Foreign key
  * A foreign key is a column or a set of columns in one table, that uniquely identifies rows in another table.

* ### 8.4. Subqueries

    ```sql
    select avg(bigscore) from
      (select max(score) as bigscore
        from mooseball
        group by team)
      as maxes; -- a table alias is required in PostgreSQL
    ```

  * [Scalar Subqueries](https://www.postgresql.org/docs/9.4/static/sql-expressions.html#SQL-SYNTAX-SCALAR-SUBQUERIES)
  * [Subquery Expressions](https://www.postgresql.org/docs/9.4/static/functions-subquery.html)
  * [FROM Clause](https://www.postgresql.org/docs/9.4/static/sql-select.html#SQL-FROM)

* ### 8.5. Views
  * A view is a select query stored in the database in a way that lets you use it like a table.
    ```sql
    create view course_size as
      select course_id, count(*) as num
        from enrollment
        group by course_id;
    ```
