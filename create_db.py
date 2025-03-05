import sqlite3

DATABASE = 'users.db'

def create_users_table():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    # Create the users table with Flask app schema if it doesn’t exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            mobile TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("Users table created or already exists with Flask schema.")

def add_otp_column():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    # Add the otp column if it doesn’t exist
    try:
        cursor.execute('ALTER TABLE users ADD COLUMN otp TEXT')
        conn.commit()
        print("OTP column added successfully.")
    except sqlite3.OperationalError as e:
        print(f"Error: {e} (Column might already exist)")
    finally:
        conn.close()

if __name__ == '__main__':
    create_users_table()  # Create the table first
    add_otp_column()      # Then add the otp column if needed
    print("Database setup completed!")