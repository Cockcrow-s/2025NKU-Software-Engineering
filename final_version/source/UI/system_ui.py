# system_ui.py

from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtCore import Qt

class SystemManagementWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("系统管理界面")
        self.setGeometry(150, 150, 400, 300)

        label = QLabel("这里是系统管理界面", self)
        label.setAlignment(Qt.AlignCenter)
        label.setGeometry(100, 120, 200, 40)