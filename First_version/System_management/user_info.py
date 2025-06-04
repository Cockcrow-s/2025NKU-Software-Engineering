import sqlite3
import hashlib
from datetime import datetime

# 连接到 SQLite 数据库（如果不存在则会自动创建）
conn = sqlite3.connect("users.db")
cursor = conn.cursor()

# 创建统一用户表
def create_users_table():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('driver', 'admin', 'mechanic')),
            register_time TEXT NOT NULL,
            last_used_time TEXT
        )
    ''')
    conn.commit()

# 基础用户类
class User:
    def __init__(self, username, password, role="driver"):
        self.username = username
        self.password = password
        self.role = role

    def register(self):
        hashed_password = hashlib.sha256(self.password.encode()).hexdigest()
        register_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            cursor.execute('''
                INSERT INTO users (username, password, role, register_time, last_used_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (self.username, hashed_password, self.role, register_time, None))
            conn.commit()
            print(f"{self.role} {self.username} 注册成功！")
        except sqlite3.IntegrityError:
            print(f"{self.role} {self.username} 已存在，请使用其他用户名！")

    def login(self):
        hashed_password = hashlib.sha256(self.password.encode()).hexdigest()
        cursor.execute('''
            SELECT * FROM users WHERE username = ? AND password = ?
        ''', (self.username, hashed_password))
        user = cursor.fetchone()
        if user:
            self.role = user[3]
            self.update_last_used_time()
            print(f"{self.role} {self.username} 登录成功！")
            return True
        else:
            print(f"{self.role} {self.username} 登录失败，用户名或密码错误！")
            return False

    def update_last_used_time(self):
        last_used_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            UPDATE users SET last_used_time = ? WHERE username = ?
        ''', (last_used_time, self.username))
        conn.commit()

    def change_password(self, old_password, new_password):
        # 对比原密码是否正确
        hashed_old = hashlib.sha256(old_password.encode()).hexdigest()
        cursor.execute('''
            SELECT password FROM users WHERE username = ?
        ''', (self.username,))
        result = cursor.fetchone()

        if not result:
            print(f"用户 {self.username} 不存在！")
            return False

        if hashed_old != result[0]:
            print("原密码错误，无法修改密码！")
            return False

        # 更新为新密码
        hashed_new = hashlib.sha256(new_password.encode()).hexdigest()
        cursor.execute('''
            UPDATE users SET password = ? WHERE username = ?
        ''', (hashed_new, self.username))
        conn.commit()
        print("密码修改成功！")
        return True

# 子类角色（继承自 User，仅设定默认角色名）
class Admin(User):
    def __init__(self, username, password):
        super().__init__(username, password, role="admin")

    def add_user(self, username, password, role):
        new_user = User(username, password, role)
        new_user.register()

class Mechanic(User):
    def __init__(self, username, password):
        super().__init__(username, password, role="mechanic")

class Driver(User):
    def __init__(self, username, password):
        super().__init__(username, password, role="driver")

# 初始化数据库函数
def initialize_user_database():
    create_users_table()

    # 检查是否已有用户
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]

    if count == 0:
        # 添加一个默认管理员
        default_admin_username = "admin"
        default_admin_password = "admin123"
        hashed_password = hashlib.sha256(default_admin_password.encode()).hexdigest()
        register_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute('''
            INSERT INTO users (username, password, role, register_time, last_used_time)
            VALUES (?, ?, ?, ?, ?)
        ''', (default_admin_username, hashed_password, "admin", register_time, None))
        conn.commit()

        print("✅ 默认管理员账号已创建：用户名 admin / 密码 admin123")

# 关闭数据库连接
def close_connection():
    conn.close()
