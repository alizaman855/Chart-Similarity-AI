import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta

def verify_credentials(username, password):
    conn = sqlite3.connect('admin.db')
    cursor = conn.cursor()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute('SELECT username FROM admin_users WHERE username = ? AND password_hash = ?', 
                   (username, password_hash))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def create_session(username):
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(hours=24)
    
    conn = sqlite3.connect('admin.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO admin_sessions (token, username, expires_at) VALUES (?, ?, ?)',
                   (token, username, expires_at))
    conn.commit()
    conn.close()
    return token

def verify_session(token):
    conn = sqlite3.connect('admin.db')
    cursor = conn.cursor()
    cursor.execute('SELECT username FROM admin_sessions WHERE token = ? AND expires_at > ?',
                   (token, datetime.now()))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None