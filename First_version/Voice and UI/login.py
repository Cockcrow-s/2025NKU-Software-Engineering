# login.py
import sys
import os

from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QMessageBox
)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'System_management')))
from user_info import User,initialize_user_database

class LoginWindow(QWidget):
    # 定义登录成功信号
    login_success = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("车载多模态智能交互系统")
        self.setFixedSize(400, 300)
        self.setupUI()
        self.create_database()
        
    def create_database(self):
        # 初始化用户数据库
        initialize_user_database()

    def setupUI(self):
        font = QFont("Arial", 10)  

        # 用户名标签和输入框
        label_user = QLabel("用户名:")
        label_user.setFont(font)
        self.edit_user = QLineEdit()
        self.edit_user.setFont(font)

        # 密码标签和输入框
        label_pass = QLabel("密码:")
        label_pass.setFont(font)
        self.edit_pass = QLineEdit()
        self.edit_pass.setEchoMode(QLineEdit.Password)
        self.edit_pass.setFont(font)

        # 登录按钮
        btn_login = QPushButton("登录")
        btn_login.setFont(font)
        btn_login.clicked.connect(self.handleLogin)

        # 布局设置
        layout = QVBoxLayout()
        layout.addWidget(label_user)
        layout.addWidget(self.edit_user)
        layout.addWidget(label_pass)
        layout.addWidget(self.edit_pass)
        layout.addWidget(btn_login)
        self.setLayout(layout)

    def handleLogin(self):
        username = self.edit_user.text().strip()
        password = self.edit_pass.text().strip()

        if not username or not password:
            QMessageBox.warning(self, "输入错误", "请输入用户名和密码")
            return

        user = User(username, password)
        if user.login():
            QMessageBox.information(self, "登录成功", f"欢迎，{username}！")
            # self.main_window = MainWindow()
            # self.main_window.show()
            self.close()
        else:
            QMessageBox.warning(self, "登录失败", "用户名或密码错误！")