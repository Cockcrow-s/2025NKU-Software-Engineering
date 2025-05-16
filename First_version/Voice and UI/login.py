# login.py
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QMessageBox
from PyQt5.QtCore import pyqtSignal

class LoginWindow(QWidget):
    # 定义登录成功信号
    login_success = pyqtSignal()

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
        user = self.edit_user.text()
        pw = self.edit_pass.text()
        # 修改为真实登录逻辑
        if user == "admin" and pw == "123":
            self.login_success.emit()  # 发出登录成功信号
            self.close()
        else:
            QMessageBox.warning(self, "错误", "用户名或密码错误")
