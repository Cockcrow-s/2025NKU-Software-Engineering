import os
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QMediaPlaylist
from PyQt5.QtCore import QUrl
import gesture_thread
# import state

class MusicPlayer:
    def add_to_playlist(self, file_path):
        """
        添加新音乐到播放列表：
        - 传入文件路径，添加到当前播放列表的末尾。
        """
        url = QUrl.fromLocalFile(file_path)
        self.playlist.addMedia(QMediaContent(url))


    def __init__(self, music_dir='music'):
        # 创建播放器和播放列表
        self.player = QMediaPlayer()
        self.playlist = QMediaPlaylist() 
        self.state = "stopped"  
        self.temp_volume = 10  # 默认音量为10%
        self.player.setVolume(self.temp_volume)

        self.gesture_use = False  # 手势识别是否继续使用

        self.gesture = gesture_thread.GestureThread()
        self.gesture.gesture_detected.connect(self.handle_gesture)

        # 加载 music_dir 目录下的所有音频文件
        for filename in os.listdir(music_dir):
            if filename.lower().endswith(('.mp3', '.wav', '.ogg', '.flac')):
                file_path = os.path.join(music_dir, filename)
                self.add_to_playlist(file_path)

        # 设置播放列表的初始索引和循环模式
        self.playlist.setCurrentIndex(0)
        self.playlist.setPlaybackMode(QMediaPlaylist.Loop)

        # 将播放列表绑定到播放器
        self.player.setPlaylist(self.playlist)

    # def load(music_dir='music'):
    #     url = QUrl.fromLocalFile(music_dir)
    #     self.playlist.addMedia(QMediaContent(url))
        
    def play(self):
        """
        播放或恢复播放：
        - 如果当前是暂停状态，调用 play() 会从暂停的位置继续；
        - 如果当前未开始播放，则从当前列表索引的开头开始播放。
        """
        self.player.play()
        self.state = "playing"

    def pause(self):
        """暂停播放，保留当前位置以便下次 play() 恢复"""
        self.player.pause()
        self.state = "stopped"
        # state.is_playing = False

    def stop(self):
        """
        停止播放并重置到列表开头：
        - 停止后，再次调用 play() 会从列表第一个文件开始播放。
        """
        self.player.stop()
        self.state = "stopped"
        self.playlist.setCurrentIndex(0)
    
    def next(self):
        """
        跳转到下一首音乐：
        - 如果当前是最后一首，则循环回到第一首。
        """
        if self.playlist.currentIndex() < self.playlist.mediaCount() - 1:
            self.playlist.next()
        else:
            self.playlist.setCurrentIndex(0)
        self.player.play()
        
    def previous(self):
        """
        跳转到上一首音乐：
        - 如果当前是第一首，则循环回到最后一首。
        """
        if self.playlist.currentIndex() > 0:
            self.playlist.previous()
        else:
            self.playlist.setCurrentIndex(self.playlist.mediaCount() - 1)
        self.player.play()
        
    def get_duration(self):
        return self.player.duration()
    
    def change_volume(self, volume):
        """
        调整音量：
        - volume: 0-100 的整数，表示音量百分比。
        """
        self.temp_volume = volume
        if volume < 0 or volume > 100:
            raise ValueError("Volume must be between 0 and 100")
        self.player.setVolume(volume)

    def change_position(self, position):
        """
        跳转到指定位置：
        - position: 以毫秒为单位的时间戳。
        """
        self.player.setPosition(position)

    def get_process(self):
        """
        获取当前播放进度：
        - 返回当前播放位置的毫秒数。
        """
        return self.player.position()
    
    def gesture_start(self):
        """
        启动手势识别线程：
        - 开始监听手势识别事件。
        """
        if not self.gesture.isRunning():
            self.gesture.start()
        
    def gesture_stop(self):
        """
        停止手势识别线程：
        - 停止监听手势识别事件。
        """
        if self.gesture.isRunning():
            self.gesture.stop()    

    def handle_gesture(self, gesture_name):
        """
        处理手势识别结果：
        - 根据手势名称执行相应的音乐控制操作。
        """
        print(f"检测到手势: {gesture_name}")
        print(f"当前状态: {self.state}")
        if gesture_name == "fist" and self.state == "playing":
            self.pause()
            self.gesture_use = True
        elif gesture_name == "thumb_up":
            self.change_volume(100)
            self.gesture_use = True
        elif gesture_name == "unknown":
            print("Error gesture.")
        elif gesture_name == "wave_hand":
            self.next()
            self.gesture_use = True

        if self.gesture_use:
            self.gesture_stop()  # 停止手势识别线程
            self.gesture_use = False          
        # 可以添加更多手势处理逻辑
