# music_window.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from music_player import MusicPlayer

class MusicWindow(QWidget):
    def __init__(self, player: MusicPlayer):
        super().__init__()
        self.setWindowTitle("音乐播放")
        self.setFixedSize(400, 180)
        self.player = player

        layout = QVBoxLayout(self)

        ctrl_layout = QHBoxLayout()
        btn_play = QPushButton("播放/继续")
        btn_play.clicked.connect(lambda: self.player.play())  # resume
        ctrl_layout.addWidget(btn_play)

        btn_pause = QPushButton("暂停")
        btn_pause.clicked.connect(self.player.pause)
        ctrl_layout.addWidget(btn_pause)

        layout.addLayout(ctrl_layout)