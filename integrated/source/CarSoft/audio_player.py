# audio_player.py
import sys
import time
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import play
import warnings

# 配置参数
AUDIO_DIRS = [
    Path("CarSoft/feedback1_audio"),
    Path("CarSoft/feedback2_audio"),
    Path("CarSoft/error_audio")
]

warnings.filterwarnings("ignore", category=RuntimeWarning)

class AudioPlayer:
    def __init__(self):
        self.playback_order = []

    def _get_audio_files(self):
        """获取所有音频文件并按时间排序"""
        all_files = []
        for directory in AUDIO_DIRS:
            if directory.exists():
                all_files.extend(directory.glob("*.wav"))
        
        # 按最后修改时间排序
        return sorted(all_files, key=lambda f: f.stat().st_mtime)

    def play_all(self):
        """播放所有音频文件"""
        files = self._get_audio_files()
        if not files:
            print("没有找到可播放的音频文件")
            return

        print(f"找到 {len(files)} 个音频文件，开始播放...")
        for idx, file_path in enumerate(files, 1):
            try:
                print(f"正在播放 ({idx}/{len(files)})：{file_path.name}")
                audio = AudioSegment.from_wav(file_path)
                play(audio)
                time.sleep(0.2)  # 播放间隔
            except Exception as e:
                print(f"播放失败 {file_path.name}: {str(e)}")

def play_audio():
    player = AudioPlayer()
    player.play_all()
    print("\n所有音频播放完成")
    
    # 释放音频设备资源
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        pa.terminate()
    except ImportError:
        pass  
    except Exception as e:
        print(f"释放音频设备时出现错误：{e}")

if __name__ == "__main__":
    play_audio()