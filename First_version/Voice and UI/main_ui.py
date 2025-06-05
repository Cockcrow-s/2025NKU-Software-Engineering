import sys
import os

from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea, QToolButton, QFileDialog, QMessageBox, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QTimer, QSize, QUrl

from music_player import MusicPlayer
from voice_thread import VoiceThread
from gesture_thread import GestureThread
from face_thread import FaceThread

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QScrollArea, QToolButton, QFileDialog, QMessageBox, QPushButton
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QTimer, QSize

from music_player import MusicPlayer
from voice_thread import VoiceThread
from gesture_thread import GestureThread
from face_thread import FaceThread
from music_window import MusicWindow

from settings_ui import SettingsWindow  # 导入设置界面

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'System_management')))
from user_info import User,initialize_user_database

class MainWindow(QMainWindow):
    def __init__(self,user):
        super().__init__()
        self.user = user
        self.setWindowTitle("车载多模态智能交互系统")
        self.setFixedSize(1024, 600)
        self.music_player = MusicPlayer()
        #语音线程
        self.voice_thread = VoiceThread(self.music_player)
        self.voice_thread.start()

        # 中央部件和布局
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        # 设置背景图片
        central_widget.setStyleSheet("background-image: url(resources/background.jpg);")
        main_layout = QVBoxLayout(central_widget)

        # 顶部信息
        top_layout = QHBoxLayout()
        self.info_icon = QLabel()
        self.info_icon.setPixmap(QIcon("resources/icons/car_status.png").pixmap(120, 80))
        top_layout.addWidget(self.info_icon)
        self.info_label = QLabel("车辆状态: 正常")
        self.warning_label = QLabel("警告!请目视前方")
        self.warning_label.setStyleSheet("color: red;")
        self.warning_icon = QLabel()
        self.warning_icon.setPixmap(QIcon("resources/icons/warning.png").pixmap(32, 32))
        top_layout.addWidget(self.info_label)
        top_layout.addStretch()
        top_layout.addWidget(self.warning_label)
        top_layout.addWidget(self.warning_icon)
        main_layout.addLayout(top_layout)

        # 滑动图标区
        scroll_area = QScrollArea()
        scroll_area.setFixedHeight(200)
        scroll_area.setWidgetResizable(True)
        icons_widget = QWidget()
        icons_layout = QHBoxLayout(icons_widget)
        icons_layout.setAlignment(Qt.AlignCenter)

        # 音乐按钮
        btn_music = QToolButton()
        btn_music.setIcon(QIcon("resources/icons/music.png"))
        btn_music.setIconSize(QSize(64, 64))
        btn_music.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn_music.setText("音乐")
        btn_music.clicked.connect(self.openMusic)
        icons_layout.addWidget(btn_music)

        # 天气按钮
        btn_weather = QToolButton()
        btn_weather.setIcon(QIcon("resources/icons/weather.png"))
        btn_weather.setIconSize(QSize(64, 64))
        btn_weather.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn_weather.setText("天气")
        btn_weather.clicked.connect(self.openWeather)
        icons_layout.addWidget(btn_weather)

        # 设置按钮
        btn_settings = QToolButton()
        btn_settings.setIcon(QIcon("resources/icons/settings.png"))
        btn_settings.setIconSize(QSize(64, 64))
        btn_settings.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn_settings.setText("设置")
        btn_settings.clicked.connect(self.openSettings)
        icons_layout.addWidget(btn_settings)

        # 系统管理按钮
        # 系统管理待完善
        btn_system = QToolButton()
        btn_system.setIcon(QIcon("resources/icons/settings.png"))
        btn_system.setIconSize(QSize(64, 64))
        btn_system.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn_system.setText("系统管理")
        btn_system.clicked.connect(self.openSystem)
        icons_layout.addWidget(btn_system)

        icons_widget.setLayout(icons_layout)
        scroll_area.setWidget(icons_widget)
        main_layout.addWidget(scroll_area)

        # 手势和面部线程

        self.gesture_thread = GestureThread()
        #self.gesture_thread.start()
        self.face_thread = FaceThread()
        #self.face_thread.start()

        # 警示灯闪烁
        self.blink_state = True
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.blinkWarning)
        self.blink_timer.start(500)

    def openMusic(self):
        # 打开音乐控制窗口
        self.music_window = MusicWindow(self.music_player)
        self.music_window.show()

    def openWeather(self):
        QMessageBox.information(self, "天气", "天气功能暂未实现")

    def openSettings(self):
        self.settings_window = SettingsWindow(self.user)
        self.settings_window.show()

    def openSystem(self):
        sys_window = QMainWindow()
        sys_window.setWindowTitle("系统管理")
        sys_window.setFixedSize(400, 300)
        sys_window.show()
        self.sys_window = sys_window

    def blinkWarning(self):
        if self.blink_state:
            self.warning_icon.hide()
        else:
            self.warning_icon.show()
        self.blink_state = not self.blink_state
