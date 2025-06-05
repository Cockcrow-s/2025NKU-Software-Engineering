import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLineEdit, QLabel, QMessageBox, QHeaderView, QAbstractItemView,QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'System_management')))
from user_info import Admin, initialize_user_database, close_connection


class SystemManagementWindow(QWidget):
    def __init__(self,user):
        super().__init__()
        self.setWindowTitle("系统管理界面")
        self.setFixedSize(1000, 750)
        self.admin =user  # 默认管理员账号
        self.setupUI()
        self.load_users()

    def setupUI(self):
        font = QFont("Arial", 10)

        # 用户信息表格
        self.table = QTableWidget()
        self.table.setFont(font)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["用户名", "角色", "注册时间"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.on_user_selected)

        # 注册新用户
        label_new_user = QLabel("新用户名:")
        label_new_user.setFont(font)
        self.edit_new_user = QLineEdit()
        self.edit_new_user.setFont(font)

        label_new_password = QLabel("新密码:")
        label_new_password.setFont(font)
        self.edit_new_password = QLineEdit()
        self.edit_new_password.setFont(font)
        self.edit_new_password.setEchoMode(QLineEdit.Password)

        label_new_role = QLabel("角色:")
        label_new_role.setFont(font)
        self.combo_new_role = QComboBox()
        self.combo_new_role.setFont(font)
        self.combo_new_role.addItems(["driver", "admin", "mechanic"])  # 下拉菜单选项

        btn_register = QPushButton("注册新用户")
        btn_register.setFont(font)
        btn_register.clicked.connect(self.register_new_user)

        # 布局
        register_layout = QVBoxLayout()
        register_layout.addWidget(label_new_user)
        register_layout.addWidget(self.edit_new_user)
        register_layout.addWidget(label_new_password)
        register_layout.addWidget(self.edit_new_password)
        register_layout.addWidget(label_new_role)
        register_layout.addWidget(self.combo_new_role)
        register_layout.addWidget(btn_register)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.table)
        main_layout.addLayout(register_layout)
       
        # 用户日志和操作区域
        self.log_area = QLabel("用户日志：")
        self.log_area.setFont(font)
        self.log_area.setWordWrap(True)
        self.log_area.setAlignment(Qt.AlignTop)

        self.btn_delete = QPushButton("注销用户")
        self.btn_delete.setFont(font)
        self.btn_delete.clicked.connect(self.delete_user)
        self.btn_delete.setEnabled(False)

        self.btn_view_logs = QPushButton("查看日志")
        self.btn_view_logs.setFont(font)
        self.btn_view_logs.clicked.connect(self.view_logs)
        self.btn_view_logs.setEnabled(False)

        # 布局
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.log_area)
        button_layout.addWidget(self.btn_delete)
        button_layout.addWidget(self.btn_view_logs)
        main_layout.addLayout(button_layout)
        main_layout.setStretchFactor(self.table, 3)  # 
        main_layout.setStretchFactor(register_layout, 1)  # 
        main_layout.setStretchFactor(button_layout, 1)
        self.setLayout(main_layout)


    def load_users(self):
        self.table.setRowCount(0)
        users = self.admin.get_all_users_info()
        for row, user in enumerate(users):
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(user["username"]))
            self.table.setItem(row, 1, QTableWidgetItem(user["role"]))
            self.table.setItem(row, 2, QTableWidgetItem(user["register_time"]))
            self.table.setItem(row, 3, QTableWidgetItem("操作"))

    def on_user_selected(self):
        selected_row = self.table.currentRow()
        if selected_row >= 0:
            self.btn_delete.setEnabled(True)
            self.btn_view_logs.setEnabled(True)
        else:
            self.btn_delete.setEnabled(False)
            self.btn_view_logs.setEnabled(False)

    def register_new_user(self):
        username = self.edit_new_user.text().strip()
        password = self.edit_new_password.text().strip()
        role = self.combo_new_role.currentText()  # 从下拉菜单获取角色
        if not username or not password:
            QMessageBox.warning(self, "注册失败", "请输入有效的用户名和密码！")
            return
        self.admin.add_user(username, password, role)
        QMessageBox.information(self, "注册成功", f"用户 {username} 注册成功！")
        self.load_users()
        self.edit_new_user.clear()
        self.edit_new_password.clear()

    def delete_user(self):
        selected_row = self.table.currentRow()
        if selected_row >= 0:
            username = self.table.item(selected_row, 0).text()
            if self.admin.delete_user(username):
                QMessageBox.information(self, "注销成功", f"用户 {username} 已注销！")
                self.load_users()
            else:
                QMessageBox.warning(self, "注销失败", f"无法注销用户 {username}！")
        else:
            QMessageBox.warning(self, "操作失败", "请选择一个用户！")

    def view_logs(self):
        selected_row = self.table.currentRow()
        if selected_row >= 0:
            username = self.table.item(selected_row, 0).text()
            logs = self.admin.get_logs_by_name(username)
            log_str = "\n".join([f"{log[0]} - {log[1]}: {log[2]} -> {log[3]}" for log in logs])
            self.log_area.setText(log_str)
        else:
            QMessageBox.warning(self, "操作失败", "请选择一个用户！")
