import mysql.connector

DB_PARAMS = {
    'database': 'employee_db',
    'user': 'root',
    'password': '8445728696',
    'host': 'localhost',
    'port': '3306'
}

try:
    conn = mysql.connector.connect(**DB_PARAMS)
    print("Connection successful!")
    conn.close()
except mysql.connector.Error as e:
    print(f"Connection failed: {e}")