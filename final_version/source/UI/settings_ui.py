# settings_ui.py
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QLabel,
    QHBoxLayout, QVBoxLayout, QStackedWidget
)
from PyQt5.QtCore import Qt

class SettingsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("系统设置")
        self.setGeometry(150, 150, 600, 400)

        # 主容器 widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 主布局（水平）
        main_layout = QHBoxLayout(main_widget)

        # 左侧菜单栏（垂直布局）
        menu_layout = QVBoxLayout()
        menu_layout.setAlignment(Qt.AlignTop)
        self.menu_buttons = []

        # 创建按钮
        self.buttons = {
            "账户": self.show_account_page,
            "个性化": self.show_personalization_page,
            "日志": self.show_logs_page,
            "系统更新": self.show_update_page
        }

        for name, callback in self.buttons.items():
            btn = QPushButton(name)
            btn.setFixedHeight(40)
            btn.clicked.connect(callback)
            menu_layout.addWidget(btn)
            self.menu_buttons.append(btn)

        # 右侧功能区域（StackedWidget）
        self.stack = QStackedWidget()
        self.stack.addWidget(self.create_placeholder_page("账户设置界面"))
        self.stack.addWidget(self.create_placeholder_page("个性化设置界面"))
        self.stack.addWidget(self.create_placeholder_page("日志界面"))
        self.stack.addWidget(self.create_placeholder_page("系统更新界面"))

        # 将左右部分加入主布局
        main_layout.addLayout(menu_layout, 1)  # 左边菜单栏（比例1）
        main_layout.addWidget(self.stack, 3)   # 右边内容区域（比例3）

    def create_placeholder_page(self, text):
        """占位页面，用于各功能页切换"""
        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        return page

    # 以下是按钮对应的切换函数
    def show_account_page(self):
        self.stack.setCurrentIndex(0)

    def show_personalization_page(self):
        self.stack.setCurrentIndex(1)

    def show_logs_page(self):
        self.stack.setCurrentIndex(2)

    def show_update_page(self):
        self.stack.setCurrentIndex(3)