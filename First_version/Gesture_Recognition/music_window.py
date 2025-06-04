from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QComboBox, QProgressBar, QFileDialog, QSlider)
from music_player import MusicPlayer
import sys  # 新增：用于处理系统退出

class MusicWindow(QWidget):
    def __init__(self, player: MusicPlayer):
        super().__init__()
        self.setWindowTitle("音乐播放")
        self.setFixedSize(400, 180)
        self.player = player

        layout = QVBoxLayout(self)

        # 控制按钮布局
        ctrl_layout = QHBoxLayout()
        btn_play = QPushButton("播放/继续")
        btn_play.clicked.connect(lambda: self.player.play())
        ctrl_layout.addWidget(btn_play)

        btn_pause = QPushButton("暂停")
        btn_pause.clicked.connect(self.player.pause)
        ctrl_layout.addWidget(btn_pause)

        # 文件选择对话框（移到按钮触发，避免启动时弹窗）
        self.btn_open = QPushButton("选择音乐")
        self.btn_open.clicked.connect(self.open_file)
        ctrl_layout.addWidget(self.btn_open)

        # 下拉框和进度条
        self.combo_music = QComboBox()
        self.process_slider = QSlider()
        self.process_slider.setOrientation(1)  # 1=水平滑块
        self.process_slider.setRange(0, 100)

        # 添加到主布局
        layout.addLayout(ctrl_layout)
        layout.addWidget(self.combo_music)
        layout.addWidget(self.process_slider)

    def open_file(self):
        """点击按钮时触发文件选择"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音乐", "", "音乐文件 (*.mp3);;所有文件 (*)"
        )
        if file_path:
            self.combo_music.addItem(file_path)
            self.player.load(file_path)  # 假设 MusicPlayer 有 load() 方法

if __name__ == "__main__":
    # 关键步骤1：创建应用实例
    app = QApplication(sys.argv)
    
    # 关键步骤2：初始化播放器和窗口
    player = MusicPlayer("C:\\Users\\jch\\Desktop\\2024.11.22良辰国风音乐会")  
    window = MusicWindow(player)
    window.show()  # 关键步骤3：显示窗口
    
    # 关键步骤4：启动事件循环
    sys.exit(app.exec_())