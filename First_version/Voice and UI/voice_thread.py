# voice_thread.py
from PyQt5.QtCore import QThread, pyqtSignal
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
import state

# 停止监听的事件
stop_event = threading.Event()

def keyword_listener(player, voice_thread_instance=None):
    samplerate, duration = 16000, 6
    frames = samplerate * duration
    temp_wav = "temp.wav"
    
    # 音频文件路径 - 如果不存在则跳过播放
    audio_dir = os.path.join(os.path.dirname(__file__), "..", "audio")
    interaction_wav = os.path.join(audio_dir, "firstinteraction.wav")
    warning_wav = os.path.join(audio_dir, "warning.wav") 
    high_warning_wav = os.path.join(audio_dir, "high_warning.wav")
    no_warning_wav = os.path.join(audio_dir, "no_warning.wav")
    
    asr = ASRExecutor()
    # 使用相对简单但稳定的模型
    model = 'conformer_wenetspeech'
    warning_count = 0  # 计数器跟踪连续警告次数
    
    print("🎤 语音关键词监听已启动...")
    
    while not stop_event.is_set():        # 检查是否处于警告状态
        if state.is_warning:
            player.pause()  # 暂停当前音乐
            if warning_count == 0:
                # 第一次警告
                print("🔊 播放警告提示音...")
                if os.path.exists(warning_wav):
                    play_specific_audio(warning_wav)
                else:
                    print("⚠️  音频文件不存在，使用文本提示：请注意行车安全")
                warning_count += 1
            else:
                # 连续警告
                print("🔊 播放高级警告提示音...")
                if os.path.exists(high_warning_wav):
                    play_specific_audio(high_warning_wav)
                else:
                    print("🚨 音频文件不存在，使用文本提示：请立即目视前方！")
                    
        print("👂 监听中...")
        rec = sd.rec(frames=frames, samplerate=samplerate, channels=1, dtype='int16', blocking=True)
        sf.write(temp_wav, rec, samplerate)
        try:
            res = asr(audio_file=temp_wav, model=model, lang='zh')
            text = res[0]['text'] if isinstance(res, list) else res
        except Exception as e:
            print("ASR 失败：", e)
            continue

        print("识别结果：", text)
          # 检查是否回应了注意道路
        if "已注意道路" in text:
            state.is_warning = False
            warning_count = 0  # 重置警告计数
            print("✅ 语音确认：已注意道路，警告解除")
            
            # 播放解除警告提示音
            if os.path.exists(no_warning_wav):
                play_specific_audio(no_warning_wav)
            else:
                print("🔔 音频文件不存在，使用文本提示：警告已解除")
                
            # 发送安全确认信号
            if voice_thread_instance:
                voice_thread_instance.safety_confirmed.emit()
            
        if "智能助手你好" in text:
            # 如果正在播放音乐，先暂停当前音乐（不修改 is_playing）
            if state.is_playing:
                player.pause()
                state.is_playing=True
            play_specific_audio(interaction_wav)
            stop_event.set()
            break

    sd.stop()
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

class VoiceThread(QThread):
    # 添加信号用于通知语音确认
    safety_confirmed = pyqtSignal()
    
    def __init__(self, player, parent=None):
        super().__init__(parent)
        self.player = player


    def run_logic(self):
        # 录音->转文字->推理->抽取->生成反馈->播放反馈
        from generate import to_record
        to_record()
        changetext()
        subprocess.run(["python", "infer.py"], check=True)
        extract_content()
        feedback_text = generate_feedback()
        play_audio()

        # 根据反馈文本控制音乐
        if "播放" in feedback_text:
            # 如果已在播放，则重新播放；否则开始播放
            self.player.play()
        if "暂停" in feedback_text:
            self.player.pause()
        ##需要实现下一首和上一首
    def run(self):
        while True:
            stop_event.clear()
            listener = threading.Thread(target=keyword_listener, args=(self.player, self))
            listener.start()
            listener.join()
            try:
                self.run_logic()
                if state.is_playing:
                    self.player.play()
            except Exception as e:
                print("逻辑执行出错：", e)
            print("循环重启监听...\n")

