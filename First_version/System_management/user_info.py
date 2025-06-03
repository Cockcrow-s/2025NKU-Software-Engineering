import sqlite3
import hashlib
from datetime import datetime

# 连接到 SQLite 数据库（如果不存在则会自动创建）
conn = sqlite3.connect("user_info.db")
cursor = conn.cursor()

# 创建用户信息表（如果尚未创建）
def create_users_table():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            register_time TEXT NOT NULL,
            last_used_time TEXT
        )
    ''')
    conn.commit()

# 创建管理员表（如果尚未创建）
def create_admin_table():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            register_time TEXT NOT NULL,
            last_used_time TEXT
        )
    ''')
    conn.commit()

# 创建车辆维修人员表（如果尚未创建）
def create_mechanic_table():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mechanics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            register_time TEXT NOT NULL,
            last_used_time TEXT
        )
    ''')
    conn.commit()

# 用户类
class User:
    def __init__(self, username, password, role="user"):
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
            print(f"用户 {self.username} 注册成功！")
        except sqlite3.IntegrityError:
            print(f"用户 {self.username} 已存在，请使用其他用户名！")

    def login(self):
        hashed_password = hashlib.sha256(self.password.encode()).hexdigest()
        cursor.execute('''
            SELECT * FROM users WHERE username = ? AND password = ?
        ''', (self.username, hashed_password))
        user = cursor.fetchone()
        if user:
            self.update_last_used_time()
            print(f"用户 {self.username} 登录成功！")
            return True
        else:
            print(f"用户 {self.username} 登录失败，用户名或密码错误！")
            return False

    def update_last_used_time(self):
        last_used_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            UPDATE users SET last_used_time = ? WHERE username = ?
        ''', (last_used_time, self.username))
        conn.commit()

# 管理员类
class Admin(User):
    def __init__(self, username, password):
        super().__init__(username, password, role="admin")

    def register(self):
        hashed_password = hashlib.sha256(self.password.encode()).hexdigest()
        register_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            cursor.execute('''
                INSERT INTO admins (username, password, register_time, last_used_time)
                VALUES (?, ?, ?, ?)
            ''', (self.username, hashed_password, register_time, None))
            conn.commit()
            print(f"管理员 {self.username} 注册成功！")
        except sqlite3.IntegrityError:
            print(f"管理员 {self.username} 已存在，请使用其他用户名！")

    def login(self):
        hashed_password = hashlib.sha256(self.password.encode()).hexdigest()
        cursor.execute('''
            SELECT * FROM admins WHERE username = ? AND password = ?
        ''', (self.username, hashed_password))
        admin = cursor.fetchone()
        if admin:
            self.update_last_used_time()
            print(f"管理员 {self.username} 登录成功！")
            return True
        else:
            print(f"管理员 {self.username} 登录失败，用户名或密码错误！")
            return False

    def update_last_used_time(self):
        last_used_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            UPDATE admins SET last_used_time = ? WHERE username = ?
        ''', (last_used_time, self.username))
        conn.commit()

    def add_admin(self, username, password):
        new_admin = Admin(username, password)
        new_admin.register()

    def add_or_modify_user(self, username, password, role="user"):
        new_user = User(username, password, role)
        new_user.register()

    def add_or_modify_mechanic(self, username, password):
        new_mechanic = Mechanic(username, password)
        new_mechanic.register()

# 车辆维修人员类
class Mechanic(User):
    def __init__(self, username, password):
        super().__init__(username, password, role="mechanic")

    def register(self):
        hashed_password = hashlib.sha256(self.password.encode()).hexdigest()
        register_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            cursor.execute('''
                INSERT INTO mechanics (username, password, register_time, last_used_time)
                VALUES (?, ?, ?, ?)
            ''', (self.username, hashed_password, register_time, None))
            conn.commit()
            print(f"车辆维修人员 {self.username} 注册成功！")
        except sqlite3.IntegrityError:
            print(f"车辆维修人员 {self.username} 已存在，请使用其他用户名！")

    def login(self):
        hashed_password = hashlib.sha256(self.password.encode()).hexdigest()
        cursor.execute('''
            SELECT * FROM mechanics WHERE username = ? AND password = ?
        ''', (self.username, hashed_password))
        mechanic = cursor.fetchone()
        if mechanic:
            self.update_last_used_time()
            print(f"车辆维修人员 {self.username} 登录成功！")
            return True
        else:
            print(f"车辆维修人员 {self.username} 登录失败，用户名或密码错误！")
            return False

    def update_last_used_time(self):
        last_used_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            UPDATE mechanics SET last_used_time = ? WHERE username = ?
        ''', (last_used_time, self.username))
        conn.commit()

# 关闭数据库连接
def close_connection():
    conn.close()

# 主程序
if __name__ == "__main__":
    create_users_table()
    create_admin_table()
    create_mechanic_table()

    # 默认创建一个管理员账号
    default_admin = Admin("admin", "admin")
    default_admin.register()

    # 示例：管理员登录
    admin = Admin("admin", "admin")
    if admin.login():
        # 管理员添加新管理员
        admin.add_admin("newadmin", "password123")
        # 管理员添加新用户
        admin.add_or_modify_user("testuser", "password123")
        # 管理员添加车辆维修人员
        admin.add_or_modify_mechanic("mechanic1", "password123")

    # 示例：用户登录
    user = User("testuser", "password123")
    user.login()

    # 示例：车辆维修人员登录
    mechanic = Mechanic("mechanic1", "password123")
    mechanic.login()

    # 关闭数据库连接
    close_connection()