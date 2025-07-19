import sqlite3
import hashlib

def init_db():
    print("Creating admin database...")
    
    conn = sqlite3.connect('admin.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin_sessions (
            token TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            expires_at TIMESTAMP NOT NULL
        )
    ''')
    
    # Create default admin user
    password_hash = hashlib.sha256('admin123'.encode()).hexdigest()
    cursor.execute('INSERT OR IGNORE INTO admin_users (username, password_hash) VALUES (?, ?)', 
                   ('admin', password_hash))
    
    conn.commit()
    conn.close()
    
    print("✅ Database created successfully!")
    print("Default credentials: admin / admin123")

if __name__ == "__main__":
    init_db()