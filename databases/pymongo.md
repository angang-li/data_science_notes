# Modify MongoDB databases with PyMongo

## 1. Database connection and basic queries

### 1.1. Connection

- Module used to connect Python with MongoDb

    ```python
    import pymongo
    ```

- Connect to local port

    ```python
    # The default port used by MongoDB is 27017
    # https://docs.mongodb.com/manual/reference/default-mongodb-port/
    conn = 'mongodb://localhost:27017'
    client = pymongo.MongoClient(conn)
    ```

- Declare the database

    ```python
    db = client.classDB
    ```

- Declare the collection

    ```python
    collection = db.classroom
    ```

### 1.2. Basic queries

- Read

    ```python
    # Query all students
    results = db.classroom.find()

    # Iterate through each student in the collection
    for student in results:
        print(student)
    ```

- Insert document

    ```python
    # Insert a document into the 'classroom' collection
    collection.insert_one(
        {
            'name': 'Ahmed',
            'row': 3,
            'favorite_python_library': 'Matplotlib',
            'hobbies': ['Running', 'Stargazing', 'Reading']
        }
    )
    ```

- Update document

    ```python
    collection.update_one(
        {'name': 'Ahmed'},
        {'$set':
            {'row': 4}
        }
    )
    ```

- Add an item to a document array

    ```python
    collection.update_one(
        {'name': 'Ahmed'},
        {'$push':
            {'hobbies': 'Listening to country music'}
        }
    )
    ```

- Delete a field from a document

    ```python
    collection.update_one({'name': 'Ahmed'},
                            {'$unset':
                            {'gavecandy': ""}
                            }
                            )
    ```

- Delete a document from a collection

    ```python
    collection.delete_one(
        {'name': 'Ahmed'}
    )
    ```