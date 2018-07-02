# SQL Alchemy

`sqlalchemy` - a Python library that provides SQL toolkit and object relational mapper (ORM)

<!-- TOC -->

- [SQL Alchemy](#sql-alchemy)
    - [1. Database connection and basic queries](#1-database-connection-and-basic-queries)
        - [1.1. Connection](#11-connection)
        - [1.2. Basic queries](#12-basic-queries)
    - [2. Create, read, update, and delete with ORM declarative base](#2-create-read-update-and-delete-with-orm-declarative-base)
        - [2.1. Create a table](#21-create-a-table)
        - [2.2. Query the table](#22-query-the-table)
        - [2.3. Add records](#23-add-records)
        - [2.4. Update records](#24-update-records)
        - [2.5. Delete records](#25-delete-records)
    - [3. Reflect and inspect existing database tables with ORM automap base](#3-reflect-and-inspect-existing-database-tables-with-orm-automap-base)
        - [3.1. Reflect database tables](#31-reflect-database-tables)
        - [3.2. Inspector](#32-inspector)

<!-- /TOC -->

## 1. Database connection and basic queries

### 1.1. Connection

- #### Dependencies

    ```python
    # SQL Alchemy
    from sqlalchemy import create_engine

    # PyMySQL 
    import pymysql # Not needed if mysqlclient is installed
    pymysql.install_as_MySQLdb()

    # Pandas
    import pandas as pd
    ```

- #### Connect engine to MySQL

    ```python
    engine = create_engine(f"mysql://{username}:{password}@{host_address}:{port}/{database_name}") # e.g. "mysql://root:password@localhost/sakila"
    conn = engine.connect()
    ```

- #### Connect engine to SQLite

    ```python
    engine = create_engine('sqlite:///pets.sqlite')
    conn = engine.connect()
    ```

### 1.2. Basic queries

- #### Query with `engine.execute`

    ```python
    data = engine.execute("SELECT * FROM Census_Data")
    records = data.fetchall() # save all records
    for record in records:
        print(record)
    ```

- #### Query with `pd.read_sql`

    ```python
    data = pd.read_sql("SELECT * FROM Census_Data", conn)
    data.head()
    ```

## 2. Create, read, update, and delete with ORM declarative base

Use declarative base when need to create class and instances

### 2.1. Create a table

- #### Dependencies

    ```python
    # Imports the method used for connecting to DBs
    from sqlalchemy import create_engine
    
    # Imports the methods needed to abstract classes into tables
    from sqlalchemy.ext.declarative import declarative_base
    
    # Sets an object to utilize the default declarative base in SQL Alchemy
    Base = declarative_base()
    
    # Import modules to declare columns and column data types
    from sqlalchemy import Column, Integer, String, Float
    ```

- #### Create a Python class

    ```python
    class Dog(Base):
        __tablename__ = 'dog'
        id = Column(Integer, primary_key=True)
        name = Column(String(255))
        color = Column(String(255))
        age = Column(Integer)
    ```

- #### Create a specific instance of the class

    ```python
    dog = Dog(name='Fido', color='Brown', age=4)
    ```

- #### Establish database connection with a `Session` object

    ```python
    engine = create_engine('sqlite:///pets.sqlite')
    conn = engine.connect()

    # Create a "Metadata" Layer That Abstracts our SQL Database
    Base.metadata.create_all(engine)

    # Create a Session Object to Connect to DB
    from sqlalchemy.orm import Session
    session = Session(bind=engine)
    ```

- #### Add records to the database

    ```python
    session.add(dog)
    session.commit()
    ```

### 2.2. Query the table

**Warning**: session is in memory, can get out of sync with database.

- #### Query

    ```python
    dog_list = session.query(Dog)
    for doggy in dog_list:
        print(doggy.name)
    ```

- #### Query with filter

    ```python
    usa = session.query(BaseballPlayer).\
            filter(BaseballPlayer.birth_country == 'USA') # can also check quantity
    ```

    ```python
    usa = session.query(BaseballPlayer).\
            filter_by(birth_country = 'USA') # filter by only checks for quality
    ```

- #### Query with limit, group_by, and aggregation

    ```python
    # Use the session to query Dow table and display the first 5 trade volumes
    for row in session.query(Dow.stock, Dow.volume).limit(15).all():
        print(row)

    # Group by
    for row in session.query(Dow.stock, Dow.volume).group_by(Dow.stock).all():
        print(row)

    # Group by and count
    from sqlalchemy import func
    tst = session.query(Demographics.location, func.count(Demographics.id)).group_by(Demographics.location).all()
    for row in tst:
        print(row)
    ```

- #### Relational join using filter

    Filter all animals from EA and NA belonging to the same sporder. This joins the data in the two tables together into a single dataset (here in the form of a tuple).

    ```python
    same_sporder = session.query(EA, NA).filter(EA.sporder == NA.sporder).limit(10).all()

    for record in same_sporder:
        (ea, na) = record
        print(ea)
        print(na)
    ```

    ```python
    sel = [EA.family, EA.genus, EA.species, NA.family, NA.genus, NA.species]
    same_sporder = session.query(*sel).filter(EA.sporder == NA.sporder).limit(10).all()

    for record in same_sporder:
        (ea_fam, ea_gen, ea_spec, na_fam, na_gen, na_spec) = record
        print(
            f"The European animal '{ea_fam} {ea_gen} {ea_spec}'"
            f"belongs to the same sporder as the North American animal '{na_fam} {na_gen} {na_spec}'.")
    ```

- #### Relational join using join

    Join all animals from EA and NA belonging to the same sporder. This joins the data in the two tables together into a single dataset (here in the form of a tuple).

    ```python
    sel = [EA.family, EA.genus, EA.species, NA.family, NA.genus, NA.species]
    same_sporder = session.query(*sel).join(EA, EA.sporder == NA.sporder).limit(10).all()

    for record in same_sporder:
        (ea_fam, ea_gen, ea_spec, na_fam, na_gen, na_spec) = record
        print(
            f"The European animal '{ea_fam} {ea_gen} {ea_spec}'"
            f"belongs to the same sporder as the North American animal '{na_fam} {na_gen} {na_spec}'.")
    ```

### 2.3. Add records

- #### Create new instances

    Note that adding to the session does not update the table. It queues up those queries.

    ```python
    session.add(Pet(name='Justin Timbersnake', type='snek', age=2))
    session.add(Pet(name='Pawtrick Stewart', type='good boy', age=10))
    session.add(Pet(name='Godzilla', type='iguana', age=1))
    session.add(Pet(name='Marshmallow', type='polar bear', age=4))
    ```

    ```python
    # The data hasn't been added yet
    engine.execute('select * from pet').fetchall()
    ```

- #### Check additions to be commited to database

    We can use the `new` attribute to see the queue of data ready to go into the database.

    ```python
    session.new
    ```

- #### Commit additions

    commit() flushes whatever remaining changes remain to the database, and commits the transaction.
    
    ```python
    session.commit()
    ```

    ```python
    # query the database
    session.query(Pet.id, Pet.name, Pet.type, Pet.age).all()
    ```

### 2.4. Update records

- #### Modify existing entries

    ```python
    # Create a query and then run update on it
    pet = session.query(Pet).filter_by(name="Marshmallow").first() # .first() to get the actual record from a set of records
    pet.age += 1
    ```

- #### Check modifications to be commited to database

    For modifications, we can use the `dirty` attribute.

    ```python
    session.dirty
    ```

- #### Commit modifications

    ```python
    session.commit()
    ```

    ```python
    # query the database
    session.query(Pet.id, Pet.name, Pet.type, Pet.age).all()
    ```

### 2.5. Delete records

- #### Delete existing entries

    ```python
    # Create a query and then delete the row collected
    pet = session.query(Pet).filter_by(id=4).delete()
    ```

- #### Commit deletion

    ```python
    session.commit()
    ```

    ```python
    # query the database
    session.query(Pet.id, Pet.name, Pet.type, Pet.age).all()
    ```

## 3. Reflect and inspect existing database tables with ORM automap base

Use automap base when the class and data already exist

### 3.1. Reflect database tables

- #### Dependencies

    ```python
    # Python SQL toolkit and Object Relational Mapper
    import sqlalchemy
    from sqlalchemy.ext.automap import automap_base
    from sqlalchemy.orm import Session
    from sqlalchemy import create_engine
    ```

- #### Engine and base

    ```python
    # Create engine using the `demographics.sqlite` database file
    engine = create_engine("sqlite:///../Resources/dow.sqlite")

    # Declare a Base using `automap_base()`
    Base = automap_base()

    # Use the Base class to reflect the database tables
    Base.prepare(engine, reflect=True)
    ```

- #### Print all of the classes mapped to the Base

    ```python
    Base.classes.keys()
    ```
- #### Query a class using Session

    ```python
    # Assign the dow class to a variable called `Dow`
    Dow = Base.classes.dow

    # Create a session
    session = Session(engine)

    # Display the row's columns and data in dictionary format
    first_row = session.query(Dow).first()
    first_row.__dict__
    ```

### 3.2. Inspector

Inspect table name, column names, and column types

- #### Dependencies

    ```python
    # Import SQLAlchemy `automap` and other dependencies
    from sqlalchemy.ext.automap import automap_base
    from sqlalchemy.orm import Session
    from sqlalchemy import create_engine, inspect
    ```

- #### Engine and inspector

    ```python
    # Create the connection engine
    engine = create_engine("sqlite:///../Resources/dow.sqlite")

    # Create the inspector and connect it to the engine
    inspector = inspect(engine)
    ```

- #### Collect the names of tables within the database

    ```python
    inspector.get_table_names()
    ```

- #### Collect the column names and types within a table

    ```python
    columns = inspector.get_columns('dow')
    for column in columns:
        print(column["name"], column["type"])
    ```
