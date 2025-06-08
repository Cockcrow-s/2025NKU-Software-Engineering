import os
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QSlider, QFileDialog, QComboBox)
from PyQt5.QtGui import QPixmap, QIcon, QTransform, QPainter, QPixmap, QRegion, QBitmap
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QRect
from music_player import MusicPlayer  
import time


class MusicWindow(QWidget):
    def __init__(self, player: MusicPlayer):
        super().__init__()
        self.setWindowTitle("音乐播放")
        self.setFixedSize(600, 800)
        self.setAutoFillBackground(False)

        # # 设置背景图,文件路径是对的，但是背景图片就是显示不出来。尝试注释了所有的控件也无济于事。
        # bg_path = os.path.join(os.path.dirname(__file__), "resources", "background", "FlierenBack.jpg").replace("\\", "/")
        # self.setObjectName("bgWindow") 
        # self.setStyleSheet(f"""
        #     QWidget#bgWindow {{
        #         background-image: url("{bg_path}");
        #         background-repeat: no-repeat;
        #         background-position: center;
        #         background-attachment: fixed;
        #     }}
        # """)

        # 图标路径
        icon_path = os.path.join(os.path.dirname(__file__), "resources", "icon", "Flieren.ico")      
        self.setWindowIcon(QIcon(icon_path))
        self.player = player

        # 当前播放状态
        self.is_playing = False
        self.player.player.durationChanged.connect(self.on_duration_changed)
        self.rotation_angle = 0
        self.duration = 0  # 默认初始值
        self.music_list = []  # 用于存储音乐列表
        # 主布局方式为纵向布局
        total_layout = QVBoxLayout(self)

        # 音乐展示区域
        music_layout = QVBoxLayout()

        # 音乐海报区域
        self.poster_label = QLabel(self)
        self.poster_label.setAlignment(Qt.AlignCenter)
        self.poster_label.setFixedSize(401, 401)
        self.poster_label.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 150px;
                background-color: #f0f0f0;
            }
        """)
        # 创建一个水平布局用于居中海报
        poster_center_layout = QHBoxLayout()
        poster_center_layout.addStretch()
        poster_center_layout.addWidget(self.poster_label)
        poster_center_layout.addStretch()

        # 音乐信息标签
        self.song_label = QLabel("未选择音乐")
        self.song_label.setFixedSize(600, 50)
        self.song_label.setAlignment(Qt.AlignCenter)
        self.song_label.setStyleSheet("font-size: 16px; font-weight: bold;")     

        # 进度条布局
        slider_layout = QHBoxLayout()
        # 进度条
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setFixedSize(400, 30)
        self.progress_slider.setRange(0, 100)
        # 通过中间槽函数处理
        self.progress_slider.sliderMoved.connect(self.on_slider_moved)
      
        # 时间标签
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFixedSize(150, 30)
        slider_layout.addWidget(self.progress_slider)
        slider_layout.addWidget(self.time_label)

        music_layout.addLayout(poster_center_layout)
        music_layout.addWidget(self.song_label)
        music_layout.addLayout(slider_layout)

        # 控制按钮布局
        ctrl_layout = QHBoxLayout()

        # 已有的音乐列表
        self.combo_music = QComboBox(self)
        self.combo_music.hide()

        # 连接信号
        btn_add = QPushButton("添加歌曲")
        btn_add.clicked.connect(
            lambda: (path := self.add_file()) and self.player.add_to_playlist(path)
        )
        ctrl_layout.addWidget(btn_add)

        btn_play = QPushButton("播放/继续")
        btn_play.clicked.connect(self.start_playing)

        ctrl_layout.addWidget(btn_play)

        btn_pause = QPushButton("暂停")
        btn_pause.clicked.connect(self.pause_playing)
        ctrl_layout.addWidget(btn_pause)

        # 文件选择对话框
        self.btn_que = QPushButton("播放队列")
        self.btn_que.clicked.connect(self.show_list)
        ctrl_layout.addWidget(self.btn_que)

        # 音量调节按钮
        volume_layout = QHBoxLayout()
        self.btn_volume = QPushButton("音量调节")
        self.btn_volume.setFixedSize(100, 30)
        self.btn_volume.clicked.connect(lambda: self.volume_slider.setVisible(not self.volume_slider.isVisible()))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setFixedSize(100, 30)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)  # 默认音量为50%
        self.volume_slider.setVisible(False)  # 初始隐藏音量滑块
        self.volume_slider.valueChanged.connect(self.player.change_volume)  # 连接音量变化信号
        volume_layout.addWidget(self.btn_volume)
        volume_layout.addWidget(self.volume_slider)

        # 添加到主布局
        total_layout.addLayout(music_layout)
        total_layout.addLayout(ctrl_layout)
        total_layout.addLayout(volume_layout)

        # 旋转动画定时器
        self.rotation_timer = QTimer(self)
        self.rotation_speed = 1  # 旋转速度(度/帧)
        self.rotation_timer.timeout.connect(self.rotate_poster)
        
        # 进度条更新定时器
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(1000)  # 每秒更新一次

    def on_slider_moved(self, value):
        """ 将滑块百分比转换为实际毫秒数 """
        duration = self.player.get_duration()
        position = int(value / 100 * duration)
        self.player.change_position(position)

    def rotate_poster(self):
        self.rotation_angle = (self.rotation_angle + self.rotation_speed) % 360
        self.update_poster_rotation()
    
    def update_poster_rotation(self):
        if not hasattr(self, "original_poster") or self.original_poster.isNull():
            return

        center = self.original_poster.rect().center()
        transform = QTransform()
        transform.translate(center.x(), center.y())
        transform.rotate(self.rotation_angle)
        transform.translate(-center.x(), -center.y())

        rotated_pixmap = self.original_poster.transformed(transform, Qt.SmoothTransformation)
        self.poster_label.setPixmap(rotated_pixmap)


    def format_time(self,ms):
        seconds = int(ms / 1000)
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def update_progress(self):
        if self.is_playing and self.duration > 0:
            current_pos = self.player.get_process() / 1000  # 当前播放位置（秒）
            progress = (current_pos / (self.duration / 1000)) * 100  # duration 是毫秒
            self.progress_slider.setValue(int(progress))
            self.update_time_label(current_pos * 1000, self.duration)  # 转回毫秒

    def update_time_label(self, current_pos, duration):
        """ 更新时间标签显示 """
        current_time = self.format_time(current_pos)
        total_time = self.format_time(duration)
        self.time_label.setText(f"{current_time} / {total_time}")

    def set_poster_and_duration_for_music(self, file_path):
        """根据音乐文件路径设置对应的海报"""
        # 获取音乐文件夹路径
        music_folder = os.path.dirname(file_path)
        # 获取文件夹名称（不带路径）
        folder_name = os.path.basename(music_folder)
        # 构造 poster 路径
        poster_path = os.path.join(music_folder, f"{folder_name}.png")
        
        if os.path.exists(poster_path):
            poster = QPixmap(poster_path)
            if not poster.isNull():
                # 将图片裁剪为圆形
                size = min(poster.width(), poster.height())  # 选择较小的边作为裁剪依据
                if size % 2 == 0:
                    size -= 1  # 确保是奇数，避免圆形裁剪时出现问题
                
                cropped_poster = poster.copy(0, 0, size, size)  # 裁剪成正方形

                # 创建圆形掩码
                mask = QBitmap(cropped_poster.size())
                mask.fill(Qt.color0)
                painter = QPainter(mask)
                painter.setBrush(Qt.color1)
                painter.setPen(Qt.color1)
                painter.drawEllipse(0, 0, size, size)
                painter.end()
                cropped_poster.setMask(mask)

                # 缩放并应用圆形效果
                scaled_poster = cropped_poster.scaled(401, 401, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.original_poster = scaled_poster  # 缓存原图
                self.poster_label.setPixmap(scaled_poster)

                # 更新音乐时长
                if hasattr(self.player, 'get_duration'):
                    duration = self.player.get_duration()
                    self.duration = duration  # 存储到实例变量
                    self.time_label.setText(f"00:00 / {self.format_time(duration)}")

                return  # 一定要 return，否则会继续执行到默认海报部分

        # 默认海报
        default_poster = QPixmap(350, 350)
        default_poster.fill(Qt.transparent)
        self.poster_label.setPixmap(default_poster)


    def add_file(self):
        """返回选择的文件路径"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音乐", "", "音乐文件 (*.mp3);;所有文件 (*)"
        )
        if file_path:
            music_name = os.path.basename(file_path).rsplit('.', 1)[0]  # 获取文件名（不含扩展名）
            self.song_label.setText(f"正在播放: {music_name}")
            self.combo_music.addItem(music_name)
            self.music_list.append(music_name)
            self.set_poster_and_duration_for_music(file_path)
            return file_path
        return None  # 明确返回None表示未选择
    
    # def play_music(self):
    #     """播放当前选中的音乐"""
    #     if not self.music_list:
    #         return
            
    #     current_index = self.combo_music.currentIndex()
    #     file_path = self.music_list[current_index]  # 获取实际文件路径
        
    #     # 更新UI
    #     music_name = os.path.basename(file_path).rsplit('.', 1)[0]
    #     self.song_label.setText(f"正在播放: {music_name}")
        
    #     # 更新海报和时长
    #     self.set_poster_and_duration_for_music(file_path)
        
    #     # 开始播放
    #     self.player.play()
    
    def show_list(self):
        """显示音乐列表"""
        if not self.combo_music.isVisible():
            self.combo_music.show()
            self.combo_music.setFixedSize(300, 30)
            self.combo_music.setStyleSheet("font-size: 14px;")
            for music in self.player.playlist:
                self.combo_music.addItem(music)
            # 连接选择事件
            self.combo_music.activated[str].connect(self.on_music_selected)
        else:
            self.combo_music.hide()

    def start_playing(self):

        self.player.play()
        self.is_playing = True
        self.rotation_timer.start(500)

    def on_duration_changed(self, duration):
        if duration > 0:
            self.duration = duration
            self.time_label.setText(f"00:00 / {self.format_time(duration)}")

    def pause_playing(self):
        self.player.pause()
        self.is_playing = False
        self.rotation_timer.stop()

if __name__ == "__main__":
    # 关键步骤1：创建应用实例
    app = QApplication(sys.argv)
    
    # 关键步骤2：初始化播放器和窗口
    initMusicDir = os.path.join(os.path.dirname(__file__), "resources", "music", "music1")
    player = MusicPlayer(initMusicDir)
    # print("播放器初始化完成，音乐目录:", initMusicDir)  
    window = MusicWindow(player)

    # 关键步骤3：显示窗口
    window.show()
    
    # 关键步骤4：启动事件循环
    sys.exit(app.exec_())