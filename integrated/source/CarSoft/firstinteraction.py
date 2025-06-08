import sys
import time
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import play
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)

def play_specific_audio(file_name="CarSoft/firstinteraction.wav"):
    """播放指定音频文件"""
    try:
        # 构建文件路径
        file_path = Path(file_name)
        
        # 检查文件是否存在
        if not file_path.exists():
            print(f"错误：音频文件 {file_name} 不存在")
            return

        print(f"开始播放音频：{file_name}")
        audio = AudioSegment.from_wav(file_path)
        play(audio)
        print("音频播放完成")

    except Exception as e:
        print(f"播放失败：{str(e)}")
    finally:
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
    target_file="CarSoft/firstinteraction.wav"
    play_specific_audio(target_file)