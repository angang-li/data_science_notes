# **Relational Database**

Virtually all relational database systems use SQL (Structured Query Language) for querying and maintaining the database.

<!-- TOC -->

- [**Relational Database**](#relational-database)
  - [1. Intro to SQL](#1-intro-to-sql)
    - [1.1. Advantages of relational databases](#11-advantages-of-relational-databases)
    - [1.2. Data types](#12-data-types)
    - [1.3. Database properties](#13-database-properties)
    - [1.4. Normalized vs denormalized table](#14-normalized-vs-denormalized-table)
    - [1.5. Tips](#15-tips)
  - [2. Create a database, table, and view](#2-create-a-database-table-and-view)
    - [2.1. Create a database](#21-create-a-database)
    - [2.2. Create a table](#22-create-a-table)
    - [2.3. View](#23-view)
  - [3. Alter, insert, update, and delete table contents](#3-alter-insert-update-and-delete-table-contents)
    - [3.1. Alter table columns](#31-alter-table-columns)
    - [3.2. Insert entries](#32-insert-entries)
    - [3.3. Update entries](#33-update-entries)
    - [3.4. Delete entries](#34-delete-entries)
  - [4. Select clauses](#4-select-clauses)
    - [4.1. Simple select](#41-simple-select)
    - [4.2. Join](#42-join)
    - [4.3. Subquery](#43-subquery)
  - [5. Write code with DB-API and command line](#5-write-code-with-db-api-and-command-line)
    - [5.1. Basic structure of DB-API](#51-basic-structure-of-db-api)
    - [5.2. Select and insert with SQLite](#52-select-and-insert-with-sqlite)
    - [5.3. Select with PostgreSQL](#53-select-with-postgresql)
    - [5.4. SQL injection attack and script injection attack](#54-sql-injection-attack-and-script-injection-attack)
    - [5.5. PostgreSQL command line](#55-postgresql-command-line)

<!-- /TOC -->


## 1. Intro to SQL

### 1.1. Advantages of relational databases

**Databases:**

- persistent storage
- safe concurrent access by multiple programs and users

**Relational databases:**

- flexible query languages with aggregation and join operations
- constraining rules for protecting consistency of data

### 1.2. Data types

- text, char(n), varchar(n)
- integer, real, double precision, decimal
- date, time, timestamp

Always write 'single quotes' around text strings and date/time values

### 1.3. Database properties

* **Atomicity**: a transaction happens as a whole or not at all
* **Consistency**: a transaction must be valid according to defined rules
* **Isolation**: read and write to multiple tables at the same time
* **Durability**: completed transactions are recorded even in the case of a system failure (e.g., power outage or crash)

### 1.4. Normalized vs denormalized table

**Rules for normalized tables:**

- Every row has the same number of columns.
- There is a unique key, and everything in a row says something about the key.
- Facts that don't relate to the key belong in different tables.
- Tables shouldn't imply relationships that don't exist.

### 1.5. Tips

- Table names are usually singular
- Column name with special characters: ``` table1.`col name @#$%7` ```

- Wildcard:
  - `'%'`: match >=1 letter, e.g., `'JO%'` -> `'JOLIAN'`, `'JOLI'`, `'JOSEPH'`
  - `'_'`: match just 1 letter, e.g., `'_AN'` -> `'DAN'`, `'MAN'`

## 2. Create a database, table, and view

### 2.1. Create a database

- #### Create and use a database

  ```sql
  DROP DATABASE IF EXISTS animals_db;
  CREATE DATABASE animals_db;
  USE animals_db;
  ```

### 2.2. Create a table

- #### Basic structure

  ```sql
  create table animals (
      name text [constraints],
      species text [constraints],
      birthdate date [constraints],
      [row constraints]);
  ```

- #### Delete a table

  ```sql
  drop table animals;
  ```

- #### Assign primary key

  Primary key must be unique.

  ```sql
  create table people (
      id  integer(11) auto_increment not null,
      first_name varchar(30) not null, 
      has_pet boolean not null,
      pet_name varchar(30),
      pet_age integer(10),
      PRIMARY KEY (id));
  ```

  ```sql
  create table animals (
      id serial primary key,
      name text,
      birthdate date);
  ```

- #### Assign multiple columns as primary key

  ```sql
  create table animals (
      postal_code text,
      country text,
      name text,
      primary key (postal_code, country));
  ```

- #### Declare relationships

  Reference provides referential integrity - columns that are supposed to refer to each other are guaranteed to do so.

  ```sql
  create table sale (
      sku text reference products (sku),
      sale_date date,
      count integer);
  ```

- #### Assign foreign key

  A foreign key is a column or a set of columns in one table, that uniquely identifies rows in another table.

  ```sql
  CREATE TABLE animals_location (
      id INTEGER(11) AUTO_INCREMENT NOT NULL,
      location VARCHAR(30) NOT NULL,
      animal_id INTEGER(10) NOT NULL,
      PRIMARY KEY (id),
      FOREIGN KEY (animal_id) REFERENCES animals_all(id));
  ```

### 2.3. View

A view is a select query stored in the database in a way that lets you use it like a table.

- #### Create a view

  ```sql
  create view course_size as
      select course_id, count(*) as num
      from enrollment
      group by course_id;
  ```
- #### Delete a view

  ```sql
  drop view course_size;
  ```

## 3. Alter, insert, update, and delete table contents

### 3.1. Alter table columns

- #### Add a column

  ```sql
  ALTER TABLE programming_languages
  ADD COLUMN new_column VARCHAR(15) DEFAULT "test" AFTER mastered;
  ```

- #### Change datatype of a column

  ```sql
  alter table actor
  modify middle_name blob;
  ```

- #### Delete a column

  ```sql
  alter table actor
  drop column middle_name;
  ```

### 3.2. Insert entries

- #### Insert one entry

  ```sql
  insert into people(first_name, has_pet, pet_name, pet_age)
  values ("Dan", true, "Rocky", 400);
  ```

- #### Insert multiple entries

  ```sql
  insert into people(first_name, has_pet, pet_name, pet_age)
  values ("Dan", true, "Rocky", 400), ("Dan", true, "Rocky", 400), ("Dan", true, "Rocky", 400);
  ```

- #### If values have same order as columns and start from the first column

  ```sql
  insert into people
  values ("Dan", true, "Rocky", 400);
  ```

### 3.3. Update entries

- #### Update entries

  ```sql
  update animals
  set name = 'cheese'
  where name like '%awful%';
  ```

### 3.4. Delete entries

- #### Delete entries

  ```sql
  SET SQL_SAFE_UPDATES = 0;

  delete from animals
  where name = 'cheese';
  
  SET SQL_SAFE_UPDATES = 1;
  ```

## 4. Select clauses

**select** *columns* **from** *tables* **where** *conditions* **;**

- Columns are separated by commas; use * to select all columns.
- The condition is Boolean (`and`, `or`, `not`). [DeMorgan's Law](https://en.wikipedia.org/wiki/De_Morgan%27s_laws) 
- Comparison operators (`=`, `!=`, `<`, `>=`, etc.)

```sql
SELECT * FROM animals_db.people;
```

is equivalent to

```sql
Use animals_db;
SELECT * FROM people;
```

### 4.1. Simple select

- #### Limit, offset

  ```sql
  select food from diet
  where species = 'llama' and name = 'Max'
  limit 10   -- 10 rows
  offset 20; -- skip 20 rows
  ```

- #### Where

  ```sql
  ... where species in ('llama', 'orangutan') ...
  ```

  ```sql
  ... where latitude between 50 and 55 ...
  ```

  ```sql
  ... where col_1 like 'pie' ... -- select where field contains words
  ```

- #### Order by

  ```sql
  select * from animals
  where species = 'orangutan'
  order by birthdate desc; -- ascending if not specified
  ```

- #### Group by with aggregations

  ```sql
  select species, min(birthdate) from animals
  group by species;
  ```

  ```sql
  select name, count(*) as num from animals
  group by name
  order by num desc
  limit 5;
  ```

- #### Having

  - `Where` is a restriction on the source tables
  - `Having` is a restriction on the results, after aggregation <br>

  ```sql
  select species, count(*) as num from animals
  group by species
  having num = 1;
  ```

### 4.2. Join

![sql join venn diagram](resources/UI25E.jpg)

- #### Inner join

  ```sql
  select animals.name, animals.species, diet.food 
  from animals
  inner join diet
  on animals.species = diet.species
  where diet.food = 'fish';
  ```

  Equivalently, without `join`

  ```sql
  select animals.name, animals.species, diet.food
  from animals, diet
  where animals.species = diet.species and diet.food = 'fish';
  ```

- #### Join on the same column name

  ```sql
  select products.name, products.sku, count(sales.sku) as num
  from products
  left join sales
  using (sku) -- equivalent to `on products.sku = sales.sku`
  ```

- #### Self Join

  ```sql
  select a.id, b.id, a.building, a.room
          from residences as a, residences as b
      where a.building = b.building
      and a.room = b.room
      and a.id < b.id
      order by a.building, a.room;
  ```

### 4.3. Subquery

- #### PostgreSQL

  ```sql
  select avg(bigscore) from
      (select max(score) as bigscore
      from mooseball
      group by team)
  as maxes; -- a table alias is required in PostgreSQL
  ```

  - [Scalar Subqueries](https://www.postgresql.org/docs/9.4/static/sql-expressions.html#SQL-SYNTAX-SCALAR-SUBQUERIES)
  - [Subquery Expressions](https://www.postgresql.org/docs/9.4/static/functions-subquery.html)
  - [FROM Clause](https://www.postgresql.org/docs/9.4/static/sql-select.html#SQL-FROM)

- #### MySQL

  Subquery after `where`

  ```sql
  SELECT *
  FROM film
  WHERE length IN (
      SELECT MAX(length), MIN(length)
      FROM film
  );
  ```
  
  Subquery after `where`, single input

  ```sql
  SELECT *
  FROM film
  WHERE length = (
      SELECT MAX(length)
      FROM film
  );
  ```

## 5. Write code with DB-API and command line

Python DB-API is a standard for python libraries that let the code to connect to the datebases.

### 5.1. Basic structure of DB-API

.connect(...) <br> &nbsp; &nbsp; &nbsp; ↓ <br>
**connection**.commit() <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; .rollback() <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; .cursor() <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ↓ <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;**cursor**.execute(query) <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;.fetchone() <br> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;.fetchall()

### 5.2. Select and insert with SQLite

- #### Select

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

- #### Insert

    ```python
    import sqlite3
    conn = sqlite3.connect('Cookies')
    cursor = conn.cursor()
    cursor.execute(
        "insert into names values ('Jennifer Smith')")
    conn.commit()
    conn.close()
    ```

### 5.3. Select with PostgreSQL

- #### Select

    ```python
    import psycopg2
    psycopg2.connect("dbname=bears")
    ```

### 5.4. SQL injection attack and script injection attack

- #### Solve SQL injection attack

  Use query parameters instead of string substitution to execute insert query!

  ```python
  cursor.execute(
      "insert into names values (%s)", (content, ))
  ```

- #### Solve script injection attack

  - `bleach.clean(...)` see [documentation](http://bleach.readthedocs.io/en/latest/)
  - `update` command

### 5.5. PostgreSQL command line

[PostgreSQL documentation](https://www.postgresql.org/docs/9.4/static/app-psql.html)

- #### Connect to a database named `forum`

  `psql forum` <br>
  if already within psql, `\c forum`

- #### Get info about the database

  - display columns of a table named `posts`
  
    `\d posts`

  - list all the tables in the database
  
    `\dt`

  - list tables plus additional information
  
    `\dt+`

  - switch between printing tables in plain text vs HTML

    `\H`

- #### Display the contents of a table named `posts` and refresh it every two seconds

  `select * from posts \watch`

