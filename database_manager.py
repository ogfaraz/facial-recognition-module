import sqlite3
from datetime import datetime

DB_NAME = "attendance.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        date TEXT NOT NULL,
                        time TEXT NOT NULL,
                        UNIQUE(name, date))''')
    conn.commit()
    conn.close()

def log_attendance(name):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.now()
    try:
        cursor.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", 
                       (name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")))
        conn.commit()
        print(f"Logged: {name}")
    except sqlite3.IntegrityError: pass
    finally: conn.close()
