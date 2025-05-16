import threading
import sounddevice as sd
import soundfile as sf
from paddlespeech.cli.asr.infer import ASRExecutor
from speech2text import changetext
from infer import content_infer
from extract import extract_content
from feedback import generate_feedback
from audio_player import play_audio
from generate import to_record
from firstinteraction import play_specific_audio
import subprocess
import os

# 线程停止信号
stop_event = threading.Event()

# 关键词监听线程函数
def keyword_listener():
    samplerate = 16000  # 采样率16kHz
    duration = 6        # 每次录音时长2秒
    frames = samplerate * duration
    temp_wav = "temp.wav"
    interaction_wav="firstinteraction.wav"
    asr = ASRExecutor()  # PaddleSpeech 识别执行器
    while not stop_event.is_set():
        # 录制音频块（阻塞，录制结束后函数才返回）
        print("监听中")
        recording = sd.rec(frames=frames, samplerate=samplerate,
                           channels=1, dtype='int16', blocking=True)
        # 保存为临时WAV文件
        sf.write(temp_wav, recording, samplerate)
        # 调用PaddleSpeech进行语音识别
        try:
            result = asr(audio_file=temp_wav)
        except Exception as e:
            print("语音识别失败：", e)
            continue
        text = result[0]['text'] if isinstance(result, list) else result
        print("识别结果：", text)
        # 检测关键词
        if "智能助手你好" in text:
            play_specific_audio(interaction_wav)
            stop_event.set()  # 设置停止事件，退出监听循环:contentReference[oaicite:8]{index=8}
            break
    # 清理资源：停止录音流（阻塞模式下一般已停止）并删除临时文件
    sd.stop()
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

# 运行给定的逻辑流程
def run_logic():
    to_record()
    changetext()
    # 运行infer.py脚本
    subprocess.run(["python", "infer.py"])
    extract_content()
    generate_feedback()
    play_audio()

if __name__ == "__main__":
    while True:
        # 启动监听线程
        stop_event.clear()
        listener_thread = threading.Thread(target=keyword_listener)
        listener_thread.start()
        # 等待监听线程检测到关键词并退出
        listener_thread.join()
        # 执行完整逻辑
        try:
            run_logic()
        except Exception as e:
            print("执行逻辑时出错：", e)
        # 循环继续，重新启动监听
        print("重新进入监听状态...\n")
