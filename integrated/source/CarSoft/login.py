# login.py
import sys
import os

from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QMessageBox, QFrame, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPalette, QBrush

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'System_management')))
from user_info import User, Admin, initialize_user_database

class LoginWindow(QWidget):
    # 定义登录成功信号
    login_success = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("车载多模态智能交互系统")
        self.setFixedSize(800, 500)
        self.setWindowIcon(QIcon("resources/login_ico.ico"))
        self.setupUI()
        self.create_database()
        
    def create_database(self):
        # 初始化用户数据库
        initialize_user_database()

    def setupUI(self):
        self.set_background("resources/loginbg.jpg")

        # 主布局
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(50, 50, 50, 50)

        # 占位空间，让登录框偏右
        spacer = QSpacerItem(100, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        main_layout.addItem(spacer)

        # 登录面板
        login_frame = QFrame()
        login_frame.setFixedSize(300, 350)
        login_frame.setStyleSheet("background-color: white; border-radius: 10px;")
        login_layout = QVBoxLayout(login_frame)
        login_layout.setContentsMargins(30, 30, 30, 30)
        login_layout.setSpacing(20)

        title = QLabel("账号登录")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 14, QFont.Bold)
        title.setFont(title_font)
        login_layout.addWidget(title)

        # 用户名输入
        self.edit_user = QLineEdit()
        self.edit_user.setPlaceholderText("请输入用户名")
        self.edit_user.setFont(QFont("Arial", 10))
        self.edit_user.setFixedHeight(35)
        self.edit_user.setStyleSheet("border: 1px solid gray; border-radius: 5px; padding-left: 8px;")
        login_layout.addWidget(self.edit_user)

        # 密码输入
        self.edit_pass = QLineEdit()
        self.edit_pass.setPlaceholderText("请输入密码")
        self.edit_pass.setFont(QFont("Arial", 10))
        self.edit_pass.setEchoMode(QLineEdit.Password)
        self.edit_pass.setFixedHeight(35)
        self.edit_pass.setStyleSheet("border: 1px solid gray; border-radius: 5px; padding-left: 8px;")
        login_layout.addWidget(self.edit_pass)

        # 登录按钮
        btn_login = QPushButton("登录")
        btn_login.setFont(QFont("Arial", 11))
        btn_login.setFixedHeight(35)
        btn_login.setStyleSheet("""
            QPushButton {
                background-color: #0078D7; color: white; border: none; border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005A9E;
            }
        """)
        btn_login.clicked.connect(self.handleLogin)
        login_layout.addWidget(btn_login)

        # 添加登录框到主布局
        main_layout.addWidget(login_frame)
        self.setLayout(main_layout)

    def set_background(self, image_path):
        # 设置背景图
        palette = QPalette()
        pixmap = QPixmap(image_path).scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        palette.setBrush(QPalette.Window, QBrush(pixmap))
        self.setPalette(palette)

    def handleLogin(self):
        username = self.edit_user.text().strip()
        password = self.edit_pass.text().strip()

        if not username or not password:
            QMessageBox.warning(self, "输入错误", "请输入用户名和密码")
            return

        user = User(username, password)
        if user.login():
            if user.role == "admin":
                user = Admin(username, password)
            QMessageBox.information(self, "登录成功", f"欢迎，{username}！")
            self.login_success.emit(user)
            self.close()
        else:
            QMessageBox.warning(self, "登录失败", "用户名或密码错误！")