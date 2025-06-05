# settings_ui.py
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QLabel,
    QHBoxLayout, QVBoxLayout, QStackedWidget, QLineEdit, QMessageBox
)
from PyQt5.QtCore import Qt

class SettingsWindow(QMainWindow):
    def __init__(self,user):
        super().__init__()
        self.user = user
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
        self.stack.addWidget(self.create_account_page())
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

    def create_account_page(self):
        self.account_page = QWidget()
        self.account_layout = QVBoxLayout(self.account_page)

        # 信息显示区域
        self.info_widget = QWidget()
        info_layout = QVBoxLayout(self.info_widget)

        # 用户名行
        row1 = QHBoxLayout()
        self.username_label = QLabel(f"用户名：{self.user.username}")
        row1.addWidget(self.username_label)
        row1.addStretch()
        btn_username = QPushButton("修改")
        btn_username.clicked.connect(self.show_username_edit)
        row1.addWidget(btn_username)
        info_layout.addLayout(row1)

        # 密码行
        row2 = QHBoxLayout()
        self.password_label = QLabel("密码：******")
        row2.addWidget(self.password_label)
        row2.addStretch()
        btn_password = QPushButton("修改")
        btn_password.clicked.connect(self.show_password_edit)
        row2.addWidget(btn_password)
        info_layout.addLayout(row2)

        # 注册时间
        register_time = self.user.get_register_time()
        self.time_label = QLabel(f"注册时间：{register_time}")
        info_layout.addWidget(self.time_label)

        self.account_layout.addWidget(self.info_widget)

        # 可切换的编辑区域
        self.edit_widget = QWidget()
        self.edit_layout = QVBoxLayout(self.edit_widget)
        self.account_layout.addWidget(self.edit_widget)
        self.edit_widget.hide()

        return self.account_page

    def show_username_edit(self):
        self.info_widget.hide()
        # 清除旧控件
        for i in reversed(range(self.edit_layout.count())):
            widget_to_remove = self.edit_layout.itemAt(i).widget()
            if widget_to_remove:
                self.edit_layout.removeWidget(widget_to_remove)
                widget_to_remove.deleteLater()

        back_btn = QPushButton("返回")
        back_btn.clicked.connect(self.back_to_info)
        new_username_input = QLineEdit()
        new_username_input.setPlaceholderText("输入新用户名")
        confirm_btn = QPushButton("确认修改")
        
        def confirm():
            new_name = new_username_input.text()
            if self.user.change_username(new_name):
                QMessageBox.information(self, "成功", "用户名修改成功！")
                self.username_label.setText(f"用户名：{self.user.username}")
                self.back_to_info()
            else:
                QMessageBox.warning(self, "失败", "用户名已存在或错误！")
                new_username_input.clear()

        confirm_btn.clicked.connect(confirm)
        self.edit_layout.addWidget(back_btn)
        self.edit_layout.addWidget(QLabel("新用户名："))
        self.edit_layout.addWidget(new_username_input)
        self.edit_layout.addWidget(confirm_btn)
        self.edit_widget.setLayout(self.edit_layout)
        self.edit_widget.show()

    def show_password_edit(self):
        self.info_widget.hide()
        # 清除旧控件
        for i in reversed(range(self.edit_layout.count())):
            widget_to_remove = self.edit_layout.itemAt(i).widget()
            if widget_to_remove:
                self.edit_layout.removeWidget(widget_to_remove)
                widget_to_remove.deleteLater()

        back_btn = QPushButton("返回")
        back_btn.clicked.connect(self.back_to_info)

        old_password_input = QLineEdit()
        old_password_input.setEchoMode(QLineEdit.Password)
        old_password_input.setPlaceholderText("原密码")

        new_password_input = QLineEdit()
        new_password_input.setEchoMode(QLineEdit.Password)
        new_password_input.setPlaceholderText("新密码")

        confirm_btn = QPushButton("确认修改")

        def confirm():
            old_pw = old_password_input.text()
            new_pw = new_password_input.text()
            if self.user.change_password(old_pw, new_pw):
                QMessageBox.information(self, "成功", "密码修改成功！")
                self.back_to_info()
            else:
                QMessageBox.warning(self, "失败", "原密码错误，无法修改密码")
                old_password_input.clear()
                new_password_input.clear()

        confirm_btn.clicked.connect(confirm)
        self.edit_layout.addWidget(back_btn)
        self.edit_layout.addWidget(QLabel("原密码："))
        self.edit_layout.addWidget(old_password_input)
        self.edit_layout.addWidget(QLabel("新密码："))
        self.edit_layout.addWidget(new_password_input)
        self.edit_layout.addWidget(confirm_btn)
        self.edit_widget.setLayout(self.edit_layout)
        self.edit_widget.show()

    def back_to_info(self):
        self.edit_widget.hide()
        self.info_widget.show()
