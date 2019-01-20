# **Relational Database**

Virtually all relational database systems use SQL (Structured Query Language) for querying and maintaining the database.

- [**Relational Database**](#relational-database)
  - [1. Intro to databases](#1-intro-to-databases)
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
    - [4.2. Functions](#42-functions)
    - [4.3. Grouping data](#43-grouping-data)
    - [4.4. Join](#44-join)
    - [4.5. Subquery](#45-subquery)
    - [4.6. Case clause](#46-case-clause)
    - [4.7. Stacking data from multiple tables](#47-stacking-data-from-multiple-tables)
    - [4.8. Window functions](#48-window-functions)
  - [5. Write code with DB-API and command line](#5-write-code-with-db-api-and-command-line)
    - [5.1. Basic structure of DB-API](#51-basic-structure-of-db-api)
    - [5.2. Select and insert with SQLite](#52-select-and-insert-with-sqlite)
    - [5.3. Select with PostgreSQL](#53-select-with-postgresql)
    - [5.4. SQL injection attack and script injection attack](#54-sql-injection-attack-and-script-injection-attack)
    - [5.5. PostgreSQL command line](#55-postgresql-command-line)

## 1. Intro to databases

A database is a collection of tables

### 1.1. Advantages of relational databases

- **Databases:**

    - Persistent storage
    - Safe concurrent access by multiple programs and users

- **Relational databases:**

    - Flexible query languages with aggregation and join operations
    - Constraining rules for protecting consistency of data

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

- **Rules for normalized tables:**

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
  CREATE TABLE animals (
      name TEXT [constraints],
      species TEXT [constraints],
      birthdate DATE [constraints],
      [row constraints]);
  ```

- #### Delete a table

  ```sql
  DROP TABLE animals;
  ```

- #### Assign primary key

  Primary key must be unique.

  ```sql
  CREATE TABLE people (
      id  INTEGER(11) AUTO_INCREMENT NOT NULL,
      first_name VARCHAR(30) NOT NULL,
      has_pet BOOLEAN NOT NULL,
      pet_name VARCHAR(30),
      pet_age INTEGER(10),
      PRIMARY KEY (id));
  ```

  ```sql
  CREATE TABLE animals (
      id SERIAL PRIMARY KEY,
      name TEXT,
      birthdate DATE);
  ```

- #### Assign multiple columns as primary key

  ```sql
  CREATE TABLE animals (
      postal_code TEXT,
      country TEXT,
      name TEXT,
      PRIMARY KEY (postal_code, country));
  ```

- #### Declare relationships

  Reference provides referential integrity - columns that are supposed to refer to each other are guaranteed to do so.

  ```sql
  CREATE TABLE sale (
      sku TEXT REFERENCES products (sku),
      sale_date DATE,
      count INTEGER);
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

A view is a select query stored in the database in a way that lets you use it like a table. It is a virtual table.

- #### Create a view

  ```sql
  CREATE VIEW v_course_size AS
      SELECT course_id, COUNT(*) AS num
      FROM enrollment
      GROUP BY course_id;
  ```
- #### Delete a view

  ```sql
  DROP VIEW v_course_size;
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
  ALTER TABLE actor
  MODIFY middle_name BLOB;
  ```

- #### Delete a column

  ```sql
  ALTER TABLE actor
  DROP COLUMN middle_name;
  ```

### 3.2. Insert entries

- #### Insert one entry

  ```sql
  INSERT INTO people(first_name, has_pet, pet_name, pet_age)
  VALUES ("Dan", true, "Rocky", 400);
  ```

- #### Insert multiple entries

  ```sql
  INSERT INTO people(first_name, has_pet, pet_name, pet_age)
  VALUES ("Dan", true, "Rocky", 400), ("Dan", true, "Rocky", 400), ("Dan", true, "Rocky", 400);
  ```

- #### If values have same order as columns and start from the first column

  ```sql
  INSERT INTO people
  VALUES ("Dan", true, "Rocky", 400);
  ```

### 3.3. Update entries

- #### Update entries

  ```sql
  UPDATE animals
  SET name = 'cheese'
  WHERE name LIKE '%awful%';
  ```

### 3.4. Delete entries

- #### Delete entries

  ```sql
  SET SQL_SAFE_UPDATES = 0;

  DELETE FROM animals
  WHERE name = 'cheese';
  
  SET SQL_SAFE_UPDATES = 1;
  ```

## 4. Select clauses

**SELECT** *columns* **FROM** *tables* **WHERE** *conditions* **;**

- Columns are separated by commas; use * to select all columns.
- The condition is Boolean (`and`, `or`, `not`). [DeMorgan's Law](https://en.wikipedia.org/wiki/De_Morgan%27s_laws) 
- Comparison operators (`=`, `!=`, `<`, `>=`, etc.)

```sql
SELECT * FROM animals_db.people;
```

is equivalent to

```sql
USE animals_db;
SELECT * FROM people;
```

### 4.1. Simple select

- #### Limit, offset

  ```sql
  SELECT food FROM diet
  WHERE species = 'llama' AND name = 'Max'
  LIMIT 10   -- show 10 rows
  OFFSET 20; -- skip 20 rows
  ```

  `LIMIT 10` is same as `FETCH FIRST 10 ROWS ONLY`

- #### Distinct

  Select unique values

  ```sql
  SELECT DISTINCT department
  FROM employees;
  ```

- #### As

  Rename columns

  ```sql
  SELECT salary AS "yearly salary"
  FROM employees;
  ```

- #### Where

  - In

    ```sql
    SELECT * FROM diet
    WHERE species IN ('llama', 'orangutan');
    ```

  - Between ... and ... (inclusive)

    ```sql
    SELECT * FROM diet
    WHERE latitude BETWEEN 50 AND 55;
    ```

  - Like

    ```sql
    SELECT * FROM diet
    WHERE col_1 LIKE 'pie'; -- select where field contains words
    ```

  - Not, !=, <>

    ```sql
    SELECT * FROM employees
    WHERE NOT department = 'Sports';
    -- equivalent to WHERE department != 'Sports'
    -- equivalent to WHERE department <> 'Sports
    ```

  - Is null

    Cannot use equality operators to compare NULL. E.g., `WHERE NULL = NULL` or `WHERE NULL != NULL` will return nothing. Use `IS` instead.

    ```sql
    SELECT * FROM employees
    WHERE email IS NULL;
    -- WHERE email IS NOT NULL;
    -- WHERE NOT email IS NULL;
    ```

  - Any, all can be applied in Where clause or Having clause

    ```sql
    SELECT * FROM employees
    WHERE region_id > ALL (
        SELECT region_id FROM region WHERE country = 'United States'
    )
    ```

- #### Order by

  ```sql
  SELECT * FROM animals
  WHERE species = 'orangutan'
  ORDER BY birthdate DESC; -- ascending (ASC) if not specified
  ```

### 4.2. Functions

- #### Boolean expression

  ```sql
  SELECT first_name, (salary > 140000) AS is_highly_paid
  FROM employees;
  ```

  ```sql
  SELECT department, ('Clothing' IN (department))
  FROM employees;
  ```

  ```sql
  SELECT department, (department LIKE '%oth%')
  FROM employees;
  ```

- #### String functions

    - Upper, lower case

        ```sql
        SELECT UPPER(first_name), LOWER(department), LENGTH(first_name)
        FROM employees;
        ```

    - Length

        ```sql
        SELECT LENGTH(first_name)
        FROM employees;
        ```

    - Trim white space

        ```sql
        SELECT TRIM(' abc    ')
        ```

    - || combine values of 2 columns together

        ```sql
        SELECT first_name || ' ' || last_name AS full_name
        FROM employees;
        ```

  - Substring

    ```sql
    SELECT SUBSTRING('This is test data' FROM 1 FOR 4) -- 'This'
    ```

    ```sql
    SELECT SUBSTRING('This is test data' FROM 9) -- 'test data'
    ```

  - Replace

    ```sql
    SELECT department, REPLACE(department, 'Clothing', 'Attire') AS "modified department"
    FROM departments; -- replace 'Clothing' by 'Attire'
    ```

  - Position

    ```sql
    SELECT POSITION('@' IN email)
    FROM employees;
    ```

    ```sql
    SELECT email, SUBSTRING(email, POSITION('@' IN email) + 1) AS email_domain
    FROM employees;
    ```

  - Coalesce null values

    ```sql
    SELECT email, COALESCE(email, 'NONE') as email_new
    FROM employees;
    ```

- #### Grouping functions

  - Min, max, avg, sum, count

    ```sql
    SELECT ROUND(AVG(salary))
    FROM employees;
    ```

    'Count' does not count null.

### 4.3. Grouping data

Group by accounts for Null.

- #### Group by with aggregations

  ```sql
  SELECT species, MIN(birthdate) FROM animals
  GROUP BY species, gender;
  ```

  ```sql
  SELECT name, COUNT(*) AS num FROM animals
  WHERE gender = 'F'
  GROUP BY name
  ORDER BY num DESC
  LIMIT 5;
  ```

- #### Having

  - `WHERE` is a restriction on the source tables
  - `HAVING` is a restriction on the results, after aggregation

  ```sql
  SELECT species, COUNT(*) AS num FROM animals
  GROUP BY species
  HAVING num = 1
  ORDER BY num DESC;
  ```

### 4.4. Join

- #### Join diagrams

  <img src="resources/UI25E.jpg" width=500>

- #### Inner join

  ```sql
  SELECT animals.name, animals.species, diet.food 
  FROM animals
  INNER JOIN diet
  ON animals.species = diet.species
  WHERE diet.food = 'fish';
  ```

  Equivalently, without `join`

  ```sql
  SELECT animals.name, animals.species, diet.food
  FROM animals, diet
  WHERE animals.species = diet.species AND diet.food = 'fish';
  ```

- #### Join on the same column name

  ```sql
  SELECT products.name, products.sku, COUNT(sales.sku) AS num
  FROM products
  LEFT JOIN sales
  USING (sku) -- equivalent to `on products.sku = sales.sku`
  ```

- #### Self Join

  ```sql
  SELECT a.id, b.id, a.building, a.room
          FROM residences AS a, residences AS b
      WHERE a.building = b.building
      AND a.room = b.room
      AND a.id < b.id
      ORDER BY a.building, a.room;
  ```

- #### Cartesian product and cross join

  When join is not specified, every single combination of rows between 2 tables is returned.

  ```sql
  SELECT *
  FROM employees, departments;
  ```

  ```sql
  SELECT *
  FROM employees CROSS JOIN department;
  ```

### 4.5. Subquery

- #### Aliasing

  ```sql
  SELECT e.department
  FROM employees AS e, department AS d;
  ```

- #### 'FROM' subquery, also known as inline view

  ```sql
  SELECT AVG(bigscore) FROM
      (SELECT MAX(score) AS bigscore
      FROM mooseball
      GROUP BY team)
  AS maxes; -- a table alias is required in PostgreSQL
  ```

  - [Scalar Subqueries](https://www.postgresql.org/docs/9.4/static/sql-expressions.html#SQL-SYNTAX-SCALAR-SUBQUERIES)
  - [Subquery Expressions](https://www.postgresql.org/docs/9.4/static/functions-subquery.html)
  - [FROM Clause](https://www.postgresql.org/docs/9.4/static/sql-select.html#SQL-FROM)

- #### 'WHERE' subquery

  Multiple inputs

  ```sql
  SELECT *
  FROM film
  WHERE length IN (
      SELECT MAX(length), MIN(length)
      FROM film
  );
  ```
  
  Single input

  ```sql
  SELECT *
  FROM film
  WHERE length = (
      SELECT MAX(length)
      FROM film
  );
  ```

- #### Correlated subquery

  The inner query uses information returned from the outer query. Subquery is checked for every single record in the outer query. Can be very slow.

  ```sql
  SELECT first_name, salary
  FROM employees AS e1
  WHERE salary > (
      SELECT AVG(salary)
      FROM employees AS e2
      WHERE e1.department = e2.department
  )
  ```

  ```sql
  SELECT department, 
    (SELECT MAX(salary) FROM employees e WHERE e.department = d.department)
  FROM departments as d;
  ```

### 4.6. Case clause

- #### Conditional expression

  ```sql
  SELECT first_name, salary,
  CASE
      WHEN salary < 100000 THEN 'Under paid'
      WHEN salary >= 100000 THEN 'Paid well'
      ELSE 'Unpaid'
  END AS category
  FROM employees
  ORDER BY salary DESC;
  ```

  ```sql
  SELECT SUM(CASE WHEN salary < 100000 THEN 1 ELSE 0 END) AS under_paid,
         SUM(CASE WHEN salary >= 100000 THEN 1 ELSE 0 END) AS paid_well
  FROM emloyees;
  ```

### 4.7. Stacking data from multiple tables

- #### Union

  Will eliminate duplicates

  ```sql
  SELECT department
  FROM departments
  UNION
  SELECT department
  FROM employees;
  ```

- #### Union all

  Will not eliminate duplicates

  ```sql
  SELECT department
  FROM departments
  UNION ALL
  SELECT department
  FROM employees;
  ```

- #### Except in PostgresQL

  Equivalent to Minus in some other databases (e.g. Oracle). Will eliminate duplicates

  ```sql
  SELECT department
  FROM employees
  EXCEPT
  SELECT department
  FROM departments;
  ```

### 4.8. Window functions

More efficient than correlated subqueries.

- #### Partition

  ```sql
  SELECT first_name, department,
  COUNT(*) OVER(PARTITION BY department) dept_count
  -- SUM(salary) OVER(PARTITION BY department)
  FROM employees;
  ```

- #### Running total

  ```sql
  SELECT first_name, hire_date, salary,
  SUM(salary) OVER(ORDER BY hire_date RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total_of_salaries -- window frame
  FROM employees;
  ```

- #### Adjacent sum

  ```sql
  SELECT first_name, hire_date, salary,
  SUM(salary) OVER(ORDER BY hire_date ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS running_total_of_salaries -- window frame
  FROM employees;
  ```

- #### Rank

  ```sql
  SELECT first_name, email, department, salary,
  RANK() OVER(PARTITION BY department ORDER BY salary DESC)
  FROM employees;
  ```

- #### Ntile

  ```sql
  SELECT first_name, email, department, salary,
  NTILE(5) OVER(PARTITION BY department ORDER BY salary DESC) salary_bracket -- split each department evenly into 5 groups
  FROM employees;
  ```

- #### First value and n-th value

  ```sql
  SELECT first_name, email, department, salary,
  FIRST_VALUE(salary) OVER(PARTITION BY department ORDER BY salary DESC) first_value -- use the first value for each department
  FROM employees;
  ```

  ```sql
  SELECT first_name, email, department, salary,
  NTH_VALUE(salary, 5) OVER(PARTITION BY department ORDER BY salary DESC) fifth_value
  FROM employees;
  ```

- #### Lead and lag

  ```sql
  SELECT first_name, email, department, salary,
  LEAD(salary) OVER() next_salary -- salary of the next row
  FROM employees;
  ```

  ```sql
  SELECT first_name, email, department, salary,
  LAG(salary) OVER() previous_salary -- salary of the previous row
  FROM employees;
  ```

- #### Grouping sets, rollup, and cube

  ```sql
  SELECT continent, country, city, SUM(units_sold)
  FROM sales
  GROUP BY GROUPING SETS(continent, country, city, ()); -- () requests the grand total
  ```

  Equivalently, rollup groups by each individual column and maintains the hierarchy

  ```sql
  SELECT continent, country, city, SUM(units_sold)
  FROM sales
  GROUP BY ROLLUP(continent, country, city);
  ```

  Cube groups by each combination of columns

  ```sql
  SELECT continent, country, city, SUM(units_sold)
  FROM sales
  GROUP BY CUBE(continent, country, city);
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

