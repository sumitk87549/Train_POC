
import psycopg2
from psycopg2 import sql

# Database connection parameters
DB_PARAMS = {
    "host": "localhost",
    "user": "postgres",
    "password": "0000", # As specified by user
    "dbname": "postgres" # Default DB to connect to initially
}

NEW_DB_NAME = "book_processing_db"

def create_database():
    """Create the target database if it doesn't exist."""
    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = True
    cur = conn.cursor()

    # Check if database exists
    cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (NEW_DB_NAME,))
    exists = cur.fetchone()

    if not exists:
        print(f"Creating database '{NEW_DB_NAME}'...")
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(NEW_DB_NAME)))
    else:
        print(f"Database '{NEW_DB_NAME}' already exists.")
    
    cur.close()
    conn.close()

def create_tables():
    """Create the necessary tables in the target database."""
    # Connect to the new database
    conn_params = DB_PARAMS.copy()
    conn_params["dbname"] = NEW_DB_NAME
    
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()

    # Create processed_books table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS processed_books (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        original_filename TEXT NOT NULL,
        cleaned_text TEXT,
        summary TEXT,
        translation TEXT,
        audio_data BYTEA,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    print("Creating 'processed_books' table...")
    cur.execute(create_table_query)
    conn.commit()
    
    print("Table created successfully.")
    cur.close()
    conn.close()

if __name__ == "__main__":
    try:
        create_database()
        create_tables()
        print("Database setup complete.")
    except Exception as e:
        print(f"Error setting up database: {e}")
