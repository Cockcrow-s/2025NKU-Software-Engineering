from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton
from system_ui import SystemManagementWindow  # 引入系统管理界面
from settings_ui import SettingsWindow  # 导入设置界面

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'System_management')))
from user_info import User,initialize_user_database

class MainWindow(QMainWindow):
    def __init__(self,user):
        super().__init__()
        self.user = user  # 保存当前登录用户
        self.setWindowTitle("车载多模态智能交互系统 - 主界面")
        self.setGeometry(100, 100, 800, 600)

        label = QLabel("欢迎进入主界面！", self)
        label.move(350, 280)

        self.manage_button = QPushButton("进入系统管理", self)
        self.manage_button.setGeometry(340, 330, 120, 40)
        self.manage_button.clicked.connect(self.open_system_management)

        self.system_window = None

        self.settings_button = QPushButton("设置", self)
        self.settings_button.move(700, 20)
        self.settings_button.clicked.connect(self.open_settings_window)

    def open_system_management(self):
        if not self.system_window:
            self.system_window = SystemManagementWindow()
        self.system_window.show()

    def open_settings_window(self):
        self.settings_window = SettingsWindow()
        self.settings_window.show()
