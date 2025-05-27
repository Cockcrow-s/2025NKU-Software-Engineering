import sqlite3
import hashlib
from datetime import datetime

# 连接到 SQLite 数据库（如果不存在则会自动创建）
conn = sqlite3.connect("user_info.db")
cursor = conn.cursor()

# 创建用户信息表（如果尚未创建）
def create_table():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            register_time TEXT NOT NULL,
            last_used_time TEXT
        )
    ''')
    conn.commit()

# 注册新用户
def register_user(username, password):
    try:
        # 计算密码的哈希值（简单示例使用 SHA256）
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        # 获取当前时间作为注册时间
        register_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO users (username, password, register_time, last_used_time)
            VALUES (?, ?, ?, ?)
        ''', (username, hashed_password, register_time, None))
        conn.commit()
        print(f"用户 {username} 注册成功！")
    except sqlite3.IntegrityError:
        print(f"用户 {username} 已存在，请使用其他用户名！")

# 验证登录用户
def login_user(username, password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute('''
        SELECT * FROM users WHERE username = ? AND password = ?
    ''', (username, hashed_password))
    user = cursor.fetchone()
    if user:
        # 如果登录成功，更新上次使用时间
        update_last_used_time(username)
        print(f"用户 {username} 登录成功！")
        return True
    else:
        print(f"用户 {username} 登录失败，用户名或密码错误！")
        return False

# 登录时更新上次使用时间
def update_last_used_time(username):
    last_used_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        UPDATE users SET last_used_time = ? WHERE username = ?
    ''', (last_used_time, username))
    conn.commit()

# 主程序
if __name__ == "__main__":
    create_table()

    # 示例：注册新用户
    register_user("testuser", "password123")

    # 示例：登录用户
    login_user("testuser", "password123")

    # 关闭数据库连接
    conn.close()