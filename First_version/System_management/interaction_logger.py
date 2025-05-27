import sqlite3
import os
from datetime import datetime
from collections import defaultdict

class InteractionLogger:
    def __init__(self, db_path="logs/interaction_log.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def _get_table_name(self, user_id):
        return f"log_{user_id}"

    def _ensure_user_table(self, user_id):
        """确保用户的日志表存在"""
        table = self._get_table_name(user_id)
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                modality TEXT,
                input_data TEXT,
                system_response TEXT
            );
        """)
        self.conn.commit()

    def log_interaction(self, user_id, modality, input_data, system_response):
        """记录一条交互日志"""
        self._ensure_user_table(user_id)
        table = self._get_table_name(user_id)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(
            f"INSERT INTO {table} (timestamp, modality, input_data, system_response) VALUES (?, ?, ?, ?)",
            (timestamp, modality, input_data, system_response)
        )
        self.conn.commit()

    def read_logs(self, user_id):
        """读取指定用户的全部日志"""
        self._ensure_user_table(user_id)
        table = self._get_table_name(user_id)
        self.cursor.execute(f"SELECT timestamp, modality, input_data, system_response FROM {table}")
        return self.cursor.fetchall()

    def generate_statistics(self, user_id):
        """返回指定用户各模态使用频次"""
        self._ensure_user_table(user_id)
        table = self._get_table_name(user_id)
        self.cursor.execute(f"SELECT modality, COUNT(*) FROM {table} GROUP BY modality")
        return dict(self.cursor.fetchall())

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    logger = InteractionLogger()

    # 记录几条日志
    logger.log_interaction("driver_001", "voice", "打开空调", "执行：ac_on")
    logger.log_interaction("driver_001", "gesture", "竖起大拇指", "执行：确认")
    logger.log_interaction("driver_001", "vision", "视线偏离", "提示：请目视前方")

    # 读取日志
    logs = logger.read_logs("driver_001")
    for log in logs:
        print(log)

    # 查看模态统计
    stats = logger.generate_statistics("driver_001")
    print("模态使用频率:", stats)

    logger.close()