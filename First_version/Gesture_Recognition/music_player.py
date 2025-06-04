import os
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QMediaPlaylist
from PyQt5.QtCore import QUrl
# import state

class MusicPlayer:
    def __init__(self, music_dir='music'):
        # 创建播放器和播放列表
        self.player = QMediaPlayer()
        self.playlist = QMediaPlaylist()    

        # 加载 music_dir 目录下的所有音频文件
        for filename in os.listdir(music_dir):
            if filename.lower().endswith(('.mp3', '.wav', '.ogg', '.flac')):
                file_path = os.path.join(music_dir, filename)
                url = QUrl.fromLocalFile(file_path)
                self.playlist.addMedia(QMediaContent(url))

        # 设置播放列表的初始索引和循环模式
        self.playlist.setCurrentIndex(0)
        self.playlist.setPlaybackMode(QMediaPlaylist.Loop)

        # 将播放列表绑定到播放器
        self.player.setPlaylist(self.playlist)

    def load(music_dir='music'):
        url = QUrl.fromLocalFile(file_path)
        self.playlist.addMedia(QMediaContent(url))
        
    def play(self):
        """
        播放或恢复播放：
        - 如果当前是暂停状态，调用 play() 会从暂停的位置继续；
        - 如果当前未开始播放，则从当前列表索引的开头开始播放。
        """
        self.player.play()
        # state.is_playing = True

    def pause(self):
        """暂停播放，保留当前位置以便下次 play() 恢复"""
        self.player.pause()
        # state.is_playing = False

    def stop(self):
        """
        停止播放并重置到列表开头：
        - 停止后，再次调用 play() 会从列表第一个文件开始播放。
        """
        self.player.stop()
        # state.is_playing = False
        self.playlist.setCurrentIndex(0)
    
    #def next():待实现
        
    #def previous():待实现
        

