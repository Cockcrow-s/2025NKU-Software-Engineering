from PyQt5.QtCore import QThread, pyqtSignal
import threading
import sounddevice as sd
import soundfile as sf
import sys
import os
from paddlespeech.cli.asr.infer import ASRExecutor
from speech2text import changetext
from infer import content_infer
from extract import extract_content
from feedback import generate_feedback
from audio_player import play_audio
from generate import to_record
from firstinteraction import play_specific_audio
import subprocess
import state
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from System_management.interaction_logger import InteractionLogger
stop_event = threading.Event()

def keyword_listener(player, voice_thread_instance=None):
    samplerate, duration = 16000, 6
    frames = samplerate * duration
    temp_wav = "CarSoft/temp.wav"
    interaction_wav = "CarSoft/firstinteraction.wav"
    warning_wav = "CarSoft/warning.wav"
    high_warning_wav = "CarSoft/high_warning.wav"
    no_warning_wav = "CarSoft/no_warning.wav" 
    asr = ASRExecutor()
    model = 'conformer_wenetspeech'
    warning_count = 0
    
    print("语音关键词监听已启动...")
    while not stop_event.is_set():          
        if state.is_warning:
            player.pause()  
            if warning_count == 0:
                # 第一次警告
                print("播放警告提示音...")
                if os.path.exists(warning_wav):
                    play_specific_audio(warning_wav)
                    # 记录警告日志
                    if voice_thread_instance and voice_thread_instance.logger:
                        voice_thread_instance.logger.log_interaction(
                            voice_thread_instance.current_user,
                            "安全警告",
                            "检测到分心驾驶",
                            "播放安全警告音"
                        )
                else:
                    print("音频文件不存在，使用文本提示：请注意行车安全")
                warning_count += 1
            else:
                # 连续警告
                print("播放高级警告提示音...")
                if os.path.exists(high_warning_wav):
                    play_specific_audio(high_warning_wav)
                else:
                    print("音频文件不存在，使用文本提示：请立即目视前方！")
                    
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
        if "道路" in text:
            state.is_warning = False
            warning_count = 0  # 重置警告计数
            print("语音确认：已注意道路，警告解除")
            
            # 播放解除警告提示音
            if os.path.exists(no_warning_wav):
                play_specific_audio(no_warning_wav)
            else:
                print("音频文件不存在，使用文本提示：警告已解除")
            
            # 记录安全确认日志
            if voice_thread_instance and voice_thread_instance.logger:
                voice_thread_instance.logger.log_interaction(
                    voice_thread_instance.current_user,
                    "安全确认",
                    text,
                    "警告解除"
                )
                
            # 发送安全确认信号
            if voice_thread_instance:
                voice_thread_instance.safety_confirmed.emit()
            
        if "智能助手你好" in text:
            # 如果正在播放音乐，先暂停当前音乐
            if state.is_playing:
                player.pause()
                state.is_playing = True
                
            play_specific_audio(interaction_wav)
            
            # 记录助手激活日志
            if voice_thread_instance and voice_thread_instance.logger:
                voice_thread_instance.logger.log_interaction(
                    voice_thread_instance.current_user,
                    "助手激活",
                    text,
                    "启动语音交互"
                )
            
            stop_event.set()
            break

    sd.stop()
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

class VoiceThread(QThread):
    # 添加信号用于通知语音确认
    safety_confirmed = pyqtSignal()
    
    def __init__(self, player, user=None, parent=None):
        super().__init__(parent)
        self.player = player
        self.current_user = user.username if user else "unknown"
        self.logger = InteractionLogger()

    def stop_thread(self):
        """停止线程"""
        print("停止语音识别线程...")
        self.running = False
        stop_event.set()
        self.quit()
        self.wait(1000) 
        if self.isRunning():
            self.terminate()  
            self.wait(1000)

    def run_logic(self):
        from generate import to_record
        to_record()
        changetext()
        subprocess.run(["python", "CarSoft/infer.py"], check=True)
        extract_content()
        feedback_text = generate_feedback()
        play_audio()
        if "播放" in feedback_text:
            self.player.play()
            action_result = "播放音乐" 
        if "暂停" in feedback_text:
            self.player.pause()
            action_result = "暂停音乐"
        self.logger.log_interaction(
                self.current_user,
                "语音交互",
                feedback_text,
                action_result
            )  

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