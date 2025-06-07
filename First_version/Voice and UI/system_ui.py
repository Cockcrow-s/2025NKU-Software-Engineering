import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLineEdit, QLabel, QMessageBox, QHeaderView, QAbstractItemView,QComboBox,
    QDialog, QScrollArea
)
from PyQt5.QtChart import QChartView, QChart, QBarSeries, QBarSet, QBarCategoryAxis
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPainter, QIcon
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'System_management')))
from user_info import Admin, initialize_user_database, close_connection

class LogViewerDialog(QDialog):
    def __init__(self, username, logger, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{username} 的日志记录")
        self.resize(800, 600)

        layout = QVBoxLayout(self)

        # === 顶部模态统计图 ===
        stats = logger.generate_statistics(username)
        if stats:
            bar_set = QBarSet("使用次数")
            categories = list(stats.keys())
            for key in categories:
                bar_set << stats[key]

            series = QBarSeries()
            series.append(bar_set)

            chart = QChart()
            chart.addSeries(series)
            chart.setTitle("模态使用统计")
            chart.setAnimationOptions(QChart.SeriesAnimations)

            axisX = QBarCategoryAxis()
            axisX.append(categories)
            chart.addAxis(axisX, Qt.AlignBottom)
            series.attachAxis(axisX)

            chart_view = QChartView(chart)
            chart_view.setRenderHint(QPainter.Antialiasing)
            chart_view.setFixedHeight(250)
            layout.addWidget(chart_view)
        else:
            layout.addWidget(QLabel("暂无模态使用统计"))

        # === 日志表格 ===
        logs = logger.read_logs(username)
        log_table = QTableWidget()
        log_table.setColumnCount(4)
        log_table.setHorizontalHeaderLabels(["时间", "模态", "输入内容", "系统响应"])
        log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        log_table.setRowCount(len(logs))

        for i, log in enumerate(logs):
            for j, value in enumerate(log):
                item = QTableWidgetItem(str(value))
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                log_table.setItem(i, j, item)

        log_table.setAlternatingRowColors(True)
        log_table.setMinimumHeight(300)

        layout.addWidget(log_table)

class RegisterUserDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("注册新用户")
        self.setFixedSize(300, 200)
        layout = QVBoxLayout()

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("输入用户名")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        
        role_layout = QHBoxLayout()
        role_label = QLabel("角色：")
        self.role_combo = QComboBox()
        self.role_combo.addItems(["driver", "admin", "mechanic"])
        role_layout.addWidget(role_label)
        role_layout.addWidget(self.role_combo)

        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("取消")
        self.btn_confirm = QPushButton("确认")
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_confirm)

        layout.addWidget(self.username_input)
        layout.addWidget(self.password_input)
        layout.addLayout(role_layout)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

class SystemManagementWindow(QWidget):
    def __init__(self, user):
        super().__init__()
        self.setWindowTitle("系统管理界面")
        self.setFixedSize(1000, 750)
        self.setWindowIcon(QIcon("resources/sys_ico.ico"))
        self.admin = user
        self.setupUI()
        self.load_users()

    def setupUI(self):
        font = QFont("Arial", 11)

        title = QLabel("用户信息管理")
        title.setFont(QFont("Arial", 15, QFont.Bold))
        title.setAlignment(Qt.AlignLeft)

        self.table = QTableWidget()
        self.table.setFont(font)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["用户名", "角色", "注册时间", "操作"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.btn_register_user = QPushButton("注册新用户")
        self.btn_register_user.setFont(font)
        self.btn_register_user.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: white;
                border: 1px solid gray;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        self.btn_register_user.setCursor(Qt.PointingHandCursor)
        self.btn_register_user.clicked.connect(self.show_register_dialog)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.table)
        layout.addWidget(self.btn_register_user, alignment=Qt.AlignRight)
        self.setLayout(layout)

    def load_users(self):
        users = self.admin.get_all_users_info()
        self.table.setRowCount(len(users))
        for row, user in enumerate(users):
            self.table.setItem(row, 0, QTableWidgetItem(user['username']))
            self.table.setItem(row, 1, QTableWidgetItem(user['role']))
            self.table.setItem(row, 2, QTableWidgetItem(user['register_time']))

            btn_logs = QPushButton("查看日志")
            btn_logs.setCursor(Qt.PointingHandCursor)
            btn_logs.clicked.connect(lambda _, u=user['username']: self.view_logs(u))

            btn_upgrade = QPushButton("推送升级")
            btn_upgrade.setCursor(Qt.PointingHandCursor)
            btn_upgrade.clicked.connect(self.notify_unimplemented)

            btn_delete = QPushButton("注销用户")
            btn_delete.setCursor(Qt.PointingHandCursor)
            btn_delete.clicked.connect(lambda _, u=user['username']: self.delete_user(u))

            op_layout = QHBoxLayout()
            op_layout.setContentsMargins(0, 0, 0, 0)
            op_layout.addWidget(btn_logs)
            op_layout.addWidget(btn_upgrade)
            op_layout.addWidget(btn_delete)

            op_widget = QWidget()
            op_widget.setLayout(op_layout)
            self.table.setCellWidget(row, 3, op_widget)

    def notify_unimplemented(self):
        QMessageBox.information(self, "提示", "该功能尚未实现")

    def delete_user(self, username):
        if self.admin.delete_user(username):
            QMessageBox.information(self, "注销成功", f"用户 {username} 已注销！")
            self.load_users()
        else:
            QMessageBox.warning(self, "注销失败", f"无法注销用户 {username}！")

    def view_logs(self,username):
        dialog = LogViewerDialog(username, self.admin.logger, self)
        dialog.exec_()

    def show_register_dialog(self):
        dialog = RegisterUserDialog()
        dialog.btn_cancel.clicked.connect(dialog.reject)
        dialog.btn_confirm.clicked.connect(lambda: self.register_new_user(dialog))
        dialog.exec_()

    def register_new_user(self, dialog):
        username = dialog.username_input.text().strip()
        password = dialog.password_input.text().strip()
        role = dialog.role_combo.currentText()
        if username and password:
            # 假设 self.admin.register_user 是注册方法
            if self.admin.add_user(username, password, role):
                QMessageBox.information(self, "成功", "用户注册成功")
                dialog.accept()
                self.load_users()
            else:
                QMessageBox.warning(self, "失败", "用户已存在或格式错误")
        else:
            QMessageBox.warning(self, "失败", "用户名和密码不能为空")
