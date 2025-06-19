import sounddevice as sd
import soundfile as sf
import os
import shutil
from datetime import datetime


OUTPUT_DIR = "CarSoft/audio"   
SAMPLE_RATE = 16000               
DURATION = 6                      
CHANNELS = 1                      
DEVICE = None                      

def prepare_output_dir():
    """准备输出目录，清空已有内容"""
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def record_audio():
    """执行录音并返回音频数据"""
    print(f"开始录音 {DURATION} 秒...")
    audio_data = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        device=DEVICE
    )
    sd.wait()
    return audio_data

def save_audio(audio_data):
    """保存音频为WAV文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"recording_{timestamp}.wav")
    
    sf.write(filename, audio_data, SAMPLE_RATE)
    print(f"音频已保存至：{filename}")

def to_record():
    prepare_output_dir()
    audio = record_audio()
    save_audio(audio)

if __name__ == "__main__":
    to_record()