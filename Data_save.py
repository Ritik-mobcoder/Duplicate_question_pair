import psycopg2
from DB_details import db_params


def table_create():
    create_table_query = """
    CREATE TABLE Prediction (
        id SERIAL PRIMARY KEY, 
        Question1 TEXT NOT NULL, 
        Question2 TEXT NOT NULL, 
        Results int
    );
    """
    try:
        connection = psycopg2.connect(**db_params)
        cursor = connection.cursor()
        cursor.execute(create_table_query)
        connection.commit()
        print("Table 'Prediction' created successfully.")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error while creating the table: {error}")
        
table_create()
        

    
    


