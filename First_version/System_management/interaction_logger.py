import sqlite3
import os
from datetime import datetime
from collections import defaultdict
import threading

class InteractionLogger:
    def __init__(self, db_path="logs/interaction_log.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # 使用线程本地存储来管理数据库连接
        self._local = threading.local()

    def _get_connection(self):
        """获取当前线程的数据库连接"""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.cursor = self._local.conn.cursor()
        return self._local.conn, self._local.cursor

    def _get_table_name(self, user_id):
        return f"log_{user_id}"

    def _ensure_user_table(self, user_id):
        """确保用户的日志表存在"""
        conn, cursor = self._get_connection()
        table = self._get_table_name(user_id)
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                modality TEXT,
                input_data TEXT,
                system_response TEXT
            );
        """)
        conn.commit()

    def log_interaction(self, user_id, modality, input_data, system_response):
        """记录一条交互日志"""
        self._ensure_user_table(user_id)
        conn, cursor = self._get_connection()
        table = self._get_table_name(user_id)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            f"INSERT INTO {table} (timestamp, modality, input_data, system_response) VALUES (?, ?, ?, ?)",
            (timestamp, modality, input_data, system_response)
        )
        conn.commit()

    def read_logs(self, user_id):
        """读取指定用户的全部日志"""
        self._ensure_user_table(user_id)
        conn, cursor = self._get_connection()
        table = self._get_table_name(user_id)
        cursor.execute(f"SELECT timestamp, modality, input_data, system_response FROM {table}")
        return cursor.fetchall()

    def generate_statistics(self, user_id):
        """返回指定用户各模态使用频次"""
        self._ensure_user_table(user_id)
        conn, cursor = self._get_connection()
        table = self._get_table_name(user_id)
        cursor.execute(f"SELECT modality, COUNT(*) FROM {table} GROUP BY modality")
        return dict(cursor.fetchall())

    def close(self):
        """关闭当前线程的数据库连接"""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            delattr(self._local, 'conn')
            delattr(self._local, 'cursor')
