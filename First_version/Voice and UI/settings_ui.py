# settings_ui.py
import os
import shutil

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QLabel,
    QHBoxLayout, QVBoxLayout, QStackedWidget, QLineEdit, QMessageBox,
    QTableWidget, QTableWidgetItem, QScrollArea, QHeaderView, QFileDialog, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtChart import QChart, QChartView, QBarSeries, QBarSet, QBarCategoryAxis
from PyQt5.QtGui import QPainter, QPixmap, QPalette, QBrush, QIcon, QFont

class SettingsWindow(QMainWindow):
    def __init__(self,user, main_window=None):
        super().__init__()
        self.user = user
        self.main_window = main_window  # 保存主界面引用
        self.setWindowTitle("系统设置")
        self.setFixedSize(800, 500)
        self.setWindowIcon(QIcon("resources/set_ico.ico"))
        self.setStyleSheet("background-color: #E6F0FA;")  # 淡蓝色背景

        # 主容器 widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 主布局（水平）
        main_layout = QHBoxLayout(main_widget)

        # 左侧菜单栏（垂直布局）
        menu_layout = QVBoxLayout()
        menu_layout.setAlignment(Qt.AlignTop)
        menu_layout.setSpacing(15)
        menu_layout.setContentsMargins(20, 20, 20, 20)

        # ========== 用户信息部分 ==========
        user_info_layout = QHBoxLayout()
        user_info_layout.setSpacing(10)

        # 用户头像
        avatar_label = QLabel()
        avatar_pixmap = QPixmap("resources/yh.png").scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        avatar_label.setPixmap(avatar_pixmap)
        avatar_label.setFixedSize(60, 60)

        # 用户名与身份标签
        info_text_layout = QVBoxLayout()
        username_label = QLabel(self.user.username)
        username_label.setFont(QFont("Arial", 11, QFont.Bold))
        role_label = QLabel(f"{self.user.role}")
        role_label.setFont(QFont("Arial", 10))
        info_text_layout.addWidget(username_label)
        info_text_layout.addWidget(role_label)

        user_info_layout.addWidget(avatar_label)
        user_info_layout.addLayout(info_text_layout)
        menu_layout.addLayout(user_info_layout)

        # 空间间隔
        menu_layout.addSpacing(20)

        # 创建按钮
        self.menu_buttons = []
        self.buttons = {
            "账户": self.show_account_page,
            "个性化": self.show_personalization_page,
            "日志": self.show_logs_page,
            "系统更新": self.show_update_page
        }

        icon_paths = [
            "resources/tb_ico1.ico",
            "resources/tb_ico2.ico",
            "resources/tb_ico3.ico",
            "resources/tb_ico4.ico"
        ]

        for i, (name, callback) in enumerate(self.buttons.items()):
            btn = QPushButton(f"  {name}")
            btn.setFixedHeight(40)
            btn.setFont(QFont("Arial", 10))
            btn.setIcon(QIcon(icon_paths[i]))
            btn.setIconSize(QSize(18, 18))
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    border: none;
                    background-color: transparent;
                    text-align: left;
                }
                QPushButton:hover {
                    background-color: #D0E7FF;
                    border-radius: 5px;
                }
            """)
            btn.clicked.connect(callback)
            menu_layout.addWidget(btn)
            self.menu_buttons.append(btn)

        # 右侧功能区域（StackedWidget）
        self.stack = QStackedWidget()
        self.stack.addWidget(self.create_account_page())
        self.stack.addWidget(self.create_personalization_page())
        self.stack.addWidget(self.create_logs_page())
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

        # ====== info_widget：封装界面四个部分（1~4）======
        self.info_widget = QWidget()
        info_layout = QVBoxLayout(self.info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(20)
        info_layout.setAlignment(Qt.AlignTop)

        # ===== (1) 用户头像 + 名称 + 身份 =====
        user_info_layout = QHBoxLayout()
        user_info_layout.setSpacing(15)

        avatar_label = QLabel()
        avatar_pixmap = QPixmap("resources/yh.png").scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        avatar_label.setPixmap(avatar_pixmap)
        avatar_label.setFixedSize(80, 80)

        text_layout = QVBoxLayout()
        username_label = QLabel(self.user.username)
        username_label.setFont(QFont("Arial", 14, QFont.Bold))
        role_label = QLabel(self.user.role)
        role_label.setFont(QFont("Arial", 12))

        text_layout.addWidget(username_label)
        text_layout.addWidget(role_label)
        user_info_layout.addWidget(avatar_label)
        user_info_layout.addLayout(text_layout)
        user_info_layout.addStretch()

        info_layout.addLayout(user_info_layout)

        # ===== (2) 注册时间 =====
        register_time = self.user.get_register_time()
        self.time_label = QLabel(f"注册时间：{register_time}")
        self.time_label.setFont(QFont("Arial", 10))
        info_layout.addWidget(self.time_label)

        # ===== (3) 标题 =====
        title_label = QLabel(f"更改 {self.user.username} 的账户")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        info_layout.addWidget(title_label)

        # ===== (4) 无边框按钮组 =====
        self.button_widget = QWidget()
        button_layout = QVBoxLayout(self.button_widget)
        button_layout.setSpacing(10)
        button_layout.setContentsMargins(0, 0, 0, 0)

        btn_username = QPushButton("更改账户名称")
        btn_username.setCursor(Qt.PointingHandCursor)
        btn_username.setFont(QFont("Arial", 11))
        btn_username.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
                color: #007ACC;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #E0F0FF;
                border-radius: 5px;
            }
        """)
        btn_username.clicked.connect(self.show_username_edit)

        btn_password = QPushButton("更改密码")
        btn_password.setCursor(Qt.PointingHandCursor)
        btn_password.setFont(QFont("Arial", 11))
        btn_password.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
                color: #007ACC;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #E0F0FF;
                border-radius: 5px;
            }
        """)
        btn_password.clicked.connect(self.show_password_edit)

        button_layout.addWidget(btn_username)
        button_layout.addWidget(btn_password)
        info_layout.addWidget(self.button_widget)

        # ===== 加入封装后的 info_widget 到主布局 =====
        self.account_layout.addWidget(self.info_widget)

        # ===== 编辑区域（用于编辑时替换显示）=====
        self.edit_widget = QWidget()
        self.edit_layout = QVBoxLayout(self.edit_widget)
        self.edit_layout.setContentsMargins(0, 0, 0, 0)
        self.edit_layout.setAlignment(Qt.AlignTop)
        self.account_layout.addWidget(self.edit_widget)
        self.edit_widget.hide()

        return self.account_page

    def show_username_edit(self):
        self.info_widget.hide()
        
        # 清除旧控件和布局
        for i in reversed(range(self.edit_layout.count())):
            item = self.edit_layout.itemAt(i)
            widget = item.widget()
            layout = item.layout()
            if widget:
                self.edit_layout.removeWidget(widget)
                widget.deleteLater()
            elif layout:
                self._clear_layout(layout)  # 清空子布局
                self.edit_layout.removeItem(layout)

        self.edit_layout.setAlignment(Qt.AlignTop)
        self.edit_layout.setSpacing(20)
        self.edit_layout.setContentsMargins(30, 30, 30, 30)

        # ===== (1) 顶部标题 =====
        title = QLabel(f"为 {self.user.username} 的账户键入一个新账户名")
        title.setFont(QFont("Arial", 13, QFont.Bold))
        self.edit_layout.addWidget(title)

        # ===== (2) 用户头像+信息 =====
        info_widget = QWidget()
        info_layout = QHBoxLayout(info_widget)
        info_layout.setSpacing(15)

        avatar = QLabel()
        avatar_pixmap = QPixmap("resources/yh.png").scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        avatar.setPixmap(avatar_pixmap)
        avatar.setFixedSize(60, 60)

        name_info_layout = QVBoxLayout()
        name_label = QLabel(self.user.username)
        name_label.setFont(QFont("Arial", 12, QFont.Bold))
        role_label = QLabel(self.user.role)
        role_label.setFont(QFont("Arial", 11))
        name_info_layout.addWidget(name_label)
        name_info_layout.addWidget(role_label)

        info_layout.addWidget(avatar)
        info_layout.addLayout(name_info_layout)
        info_layout.addStretch()
        self.edit_layout.addWidget(info_widget)

        # ===== (3) 输入框 =====
        new_username_input = QLineEdit()
        new_username_input.setPlaceholderText("新账户名")
        new_username_input.setFont(QFont("Arial", 11))
        new_username_input.setFixedHeight(30)
        self.edit_layout.addWidget(new_username_input)

        # ===== (4) 按钮组（靠右）=====
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        confirm_btn = QPushButton("更改名称")
        confirm_btn.setFixedSize(90, 30)
        confirm_btn.setFont(QFont("Arial", 10))

        cancel_btn = QPushButton("取消")
        cancel_btn.setFixedSize(80, 30)
        cancel_btn.setFont(QFont("Arial", 10))
        cancel_btn.clicked.connect(self.back_to_info)

        def confirm():
            new_name = new_username_input.text().strip()
            if self.user.change_username(new_name):
                QMessageBox.information(self, "成功", "用户名修改成功！")
                self.username_label.setText(f"用户名：{self.user.username}")
                self.back_to_info()
            else:
                QMessageBox.warning(self, "失败", "用户名已存在或错误！")
                new_username_input.clear()

        confirm_btn.clicked.connect(confirm)

        btn_layout.addWidget(confirm_btn)
        btn_layout.addWidget(cancel_btn)
        self.edit_layout.addLayout(btn_layout)

        # ===== 显示编辑区域 =====
        self.edit_widget.setLayout(self.edit_layout)
        self.edit_widget.show()

    def show_password_edit(self):
        self.info_widget.hide()

        # 清除旧控件和布局
        for i in reversed(range(self.edit_layout.count())):
            item = self.edit_layout.itemAt(i)
            widget = item.widget()
            layout = item.layout()
            if widget:
                self.edit_layout.removeWidget(widget)
                widget.deleteLater()
            elif layout:
                self._clear_layout(layout)
                self.edit_layout.removeItem(layout)

        self.edit_layout.setAlignment(Qt.AlignTop)
        self.edit_layout.setSpacing(20)
        self.edit_layout.setContentsMargins(30, 30, 30, 30)

        # ===== (1) 顶部标题 =====
        title = QLabel(f"更改 {self.user.username} 的密码")
        title.setFont(QFont("Arial", 13, QFont.Bold))
        self.edit_layout.addWidget(title)

        # ===== (2) 用户头像 + 信息 =====
        info_widget = QWidget()
        info_layout = QHBoxLayout(info_widget)
        info_layout.setSpacing(15)

        avatar = QLabel()
        avatar_pixmap = QPixmap("resources/yh.png").scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        avatar.setPixmap(avatar_pixmap)
        avatar.setFixedSize(60, 60)

        name_info_layout = QVBoxLayout()
        name_label = QLabel(self.user.username)
        name_label.setFont(QFont("Arial", 12, QFont.Bold))
        role_label = QLabel(self.user.role)
        role_label.setFont(QFont("Arial", 11))
        name_info_layout.addWidget(name_label)
        name_info_layout.addWidget(role_label)

        info_layout.addWidget(avatar)
        info_layout.addLayout(name_info_layout)
        info_layout.addStretch()

        self.edit_layout.addWidget(info_widget)

        # ===== (3) 输入框 =====
        old_password_input = QLineEdit()
        old_password_input.setEchoMode(QLineEdit.Password)
        old_password_input.setPlaceholderText("当前密码")
        old_password_input.setFont(QFont("Arial", 11))
        old_password_input.setFixedHeight(30)

        new_password_input = QLineEdit()
        new_password_input.setEchoMode(QLineEdit.Password)
        new_password_input.setPlaceholderText("新密码")
        new_password_input.setFont(QFont("Arial", 11))
        new_password_input.setFixedHeight(30)

        confirm_password_input = QLineEdit()
        confirm_password_input.setEchoMode(QLineEdit.Password)
        confirm_password_input.setPlaceholderText("确认新密码")
        confirm_password_input.setFont(QFont("Arial", 11))
        confirm_password_input.setFixedHeight(30)

        self.edit_layout.addWidget(old_password_input)
        self.edit_layout.addWidget(new_password_input)
        self.edit_layout.addWidget(confirm_password_input)

        # ===== (4) 按钮组 =====
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        confirm_btn = QPushButton("更改密码")
        confirm_btn.setFixedSize(90, 30)
        confirm_btn.setFont(QFont("Arial", 10))

        cancel_btn = QPushButton("取消")
        cancel_btn.setFixedSize(80, 30)
        cancel_btn.setFont(QFont("Arial", 10))
        cancel_btn.clicked.connect(self.back_to_info)

        def confirm():
            old_pw = old_password_input.text().strip()
            new_pw = new_password_input.text().strip()
            confirm_pw = confirm_password_input.text().strip()

            if new_pw != confirm_pw:
                QMessageBox.warning(self, "错误", "两次输入的新密码不一致！")
                confirm_password_input.clear()
                return

            if self.user.change_password(old_pw, new_pw):
                QMessageBox.information(self, "成功", "密码修改成功！")
                self.back_to_info()
            else:
                QMessageBox.warning(self, "失败", "原密码错误，无法修改密码")
                old_password_input.clear()
                new_password_input.clear()
                confirm_password_input.clear()

        confirm_btn.clicked.connect(confirm)

        btn_layout.addWidget(confirm_btn)
        btn_layout.addWidget(cancel_btn)
        self.edit_layout.addLayout(btn_layout)

        # ===== 显示编辑区域 =====
        self.edit_widget.show()

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget:
                widget.deleteLater()
            elif child_layout:
                self._clear_layout(child_layout)

    def back_to_info(self):
        self.edit_widget.hide()
        self.info_widget.show()

    def create_logs_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # === 顶部模态统计部分 ===
        stats = self.user.logger.generate_statistics(self.user.username)
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
            layout.addWidget(QLabel("暂无日志统计"))

        # === 下方日志表格部分 ===
        logs = self.user.logger.read_logs(self.user.username)

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

        log_table.setMinimumHeight(300)
        log_table.setAlternatingRowColors(True)
        layout.addWidget(log_table)

        return page

    def create_personalization_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # ===== (1) 顶部标题 =====
        title = QLabel("个性化设置背景")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        # ===== (2) 背景预览区域 =====
        self.preview_label = QLabel("背景预览区域")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedHeight(200)
        self.preview_label.setStyleSheet("border: 1px solid gray; background-color: #F9F9F9;")
        layout.addWidget(self.preview_label)

        # ===== (3) 选择照片行 =====
        choose_row = QHBoxLayout()
        choose_row.setSpacing(10)

        choose_label = QLabel("选择一张照片")
        choose_label.setFont(QFont("Arial", 11))

        browse_button = QPushButton("浏览照片")
        browse_button.setFixedSize(100, 30)
        browse_button.setStyleSheet("""
            QPushButton {
                background-color: white;
                border: 1px solid gray;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        browse_button.clicked.connect(self.select_background_image)

        choose_row.addWidget(choose_label)
        choose_row.addStretch()
        choose_row.addWidget(browse_button)
        layout.addLayout(choose_row)

        # ===== (4) 确认修改按钮（蓝字透明）=====
        confirm_button = QPushButton("确认修改")
        confirm_button.setFont(QFont("Arial", 11))
        confirm_button.setCursor(Qt.PointingHandCursor)
        confirm_button.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
                color: #007ACC;
            }
            QPushButton:hover {
                background-color: #E0F0FF;
                border-radius: 5px;
            }
        """)
        confirm_button.clicked.connect(self.confirm_background_change)
        layout.addWidget(confirm_button, alignment=Qt.AlignRight)

        # ===== (5) 恢复默认背景按钮 =====
        restore_button = QPushButton("恢复默认背景")
        restore_button.setFont(QFont("Arial", 10))
        restore_button.setCursor(Qt.PointingHandCursor)
        restore_button.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
                color: gray;
            }
            QPushButton:hover {
                text-decoration: underline;
            }
        """)
        restore_button.clicked.connect(self.restore_default_background)
        layout.addWidget(restore_button, alignment=Qt.AlignRight)

        return page

    def select_background_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择背景图片", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.temp_background_path = file_path  # 暂存路径，等待确认
            self.preview_label.setPixmap(QPixmap(file_path).scaled(
                self.preview_label.width(),
                self.preview_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

    def confirm_background_change(self):
        if hasattr(self, "temp_background_path") and self.temp_background_path:
            save_path = os.path.join("resources", os.path.basename(self.temp_background_path))
            os.makedirs("resources", exist_ok=True)
            shutil.copy(self.temp_background_path, save_path)

            # 通知主界面更换背景（主界面需要实现 set_background）
            if self.main_window and hasattr(self.main_window, "set_background"):
                self.main_window.set_background(save_path)

            QMessageBox.information(self, "成功", "背景已成功更改！")
        else:
            QMessageBox.warning(self, "提示", "请先选择一张照片")

    def restore_default_background(self):
        default_path = os.path.join("resources", "background.jpg")
        if os.path.exists(default_path):
            self.preview_label.setPixmap(QPixmap(default_path).scaled(
                self.preview_label.width(),
                self.preview_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            if self.main_window and hasattr(self.main_window, "set_background"):
                self.main_window.set_background(default_path)
            QMessageBox.information(self, "已恢复", "背景已恢复为默认图片")
        else:
            QMessageBox.warning(self, "错误", "默认背景图片不存在！")
