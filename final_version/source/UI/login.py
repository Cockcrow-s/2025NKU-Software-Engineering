import sys
import os
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QMessageBox
)
from PyQt5.QtCore import Qt
from ui_main import MainWindow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'System_management')))
from user_info import User

class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("车载多模态智能交互系统")
        self.setFixedSize(1024, 600)
        self.setupUI()

    def setupUI(self):
        label_user = QLabel("用户名:")
        self.edit_user = QLineEdit()
        label_pass = QLabel("密码:")
        self.edit_pass = QLineEdit()
        self.edit_pass.setEchoMode(QLineEdit.Password)
        btn_login = QPushButton("登录")
        btn_login.clicked.connect(self.handleLogin)

        layout = QVBoxLayout(self)
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
            self.main_window = MainWindow()
            self.main_window.show()
            self.close()
        else:
            QMessageBox.warning(self, "登录失败", "用户名或密码错误！")
