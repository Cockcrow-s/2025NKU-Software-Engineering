# gesture_thread.py
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import state
import cv2
from firstinteraction import play_specific_audio
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from System_management.interaction_logger import InteractionLogger

no_warning_wav = "CarSoft/no_warning.wav"

class GestureThread(QThread):
    # 定义信号
    gesture_recognized = pyqtSignal(str)  # 手势识别信号
    gesture_action = pyqtSignal(str)  # 手势动作信号
    music_control = pyqtSignal(str)  # 音乐控制信号
    
    def __init__(self, user=None, parent=None):
        super().__init__(parent)
        self.running = False
        self.last_gesture = None
        self.gesture_count = {}
        self.gesture_cooldown = 2.0
        self.last_gesture_time = 0
        self.current_user = user.username if user else "unknown"
        self.logger = InteractionLogger()
        
        print(f"手势识别线程初始化完成，用户: {self.current_user}")
        
    def start_gesture_recognition(self):
        """开始手势识别"""
        self.running = True
        self.start()
        print(f"手势识别系统启动，用户: {self.current_user}")
        
        # 记录系统启动日志
        self.logger.log_interaction(
            self.current_user,
            "系统状态",
            "手势识别系统启动",
            "手势识别线程开始运行"
        )
    
    def stop_gesture_recognition(self):
        """停止手势识别"""
        print("停止手势识别线程...")
        
        # 记录系统停止日志
        self.logger.log_interaction(
            self.current_user,
            "系统状态",
            "手势识别系统停止",
            "用户请求停止手势识别线程"
        )
        
        self.running = False
        self.quit()
        self.wait(3000)  # 等待最多3秒
        if self.isRunning():
            self.terminate()  # 强制终止
            self.wait(1000)
        print("手势识别线程已停止")
    
        
    def process_gesture(self, gesture):
        """处理从face_thread接收到的手势"""
        current_time = time.time()
        
        # 添加手势防抖处理和冷却时间
        if gesture and (gesture != self.last_gesture or 
                       current_time - self.last_gesture_time > self.gesture_cooldown):
            self.gesture_recognized.emit(gesture)
            print(f"检测到手势: {gesture}")
            
            # 记录手势识别日志
            self.logger.log_interaction(
                self.current_user,
                "手势识别",
                f"检测到手势: {gesture}",
                "手势识别成功"
            )
            
            # 处理手势动作
            if gesture == 'fist':
                # 握拳手势控制音乐播放/暂停
                self.handle_music_play_pause()
            elif gesture == 'wave':
                # 挥手手势切换音乐（在安全状态下）
                if not state.is_warning:
                    self.handle_music_next()
                else:
                    # 在警告状态下仍然表示拒绝
                    self.handle_reject_gesture(gesture)
            elif gesture in ['nod', 'thumbs_up', 'ok']:
                self.handle_confirm_gesture(gesture)
            elif gesture == 'shake':
                self.handle_reject_gesture(gesture)
            elif gesture == 'stop':
                self.handle_stop_gesture()
                
            self.last_gesture = gesture
            self.last_gesture_time = current_time
    
    def handle_music_play_pause(self):
        """处理音乐播放/暂停手势"""
        print("握拳手势: 控制音乐播放/暂停")
        
        # 确定当前音乐状态
        current_state = "播放中" if state.is_playing else "已暂停"
        target_action = "暂停音乐" if state.is_playing else "播放音乐"
        
        # 记录音乐控制日志
        self.logger.log_interaction(
            self.current_user,
            "手势控制",
            f"握拳手势控制音乐 - 当前状态: {current_state}",
            f"执行{target_action}"
        )
        
        self.music_control.emit("play_pause")
        self.gesture_action.emit("music_play_pause")
    
    def handle_music_next(self):
        """处理音乐切换手势"""
        print("挥手手势: 切换到下一首音乐")
        
        # 记录音乐切换日志
        self.logger.log_interaction(
            self.current_user,
            "手势控制",
            "挥手手势切换音乐",
            "切换到下一首音乐"
        )
        
        self.music_control.emit("next")
        self.gesture_action.emit("music_next")
    
    def handle_confirm_gesture(self, gesture):
        """处理确认类手势（点头、大拇指、OK手势）"""
        print(f"确认手势: {gesture}")
        
        if state.is_warning:
            # 如果当前在警告状态，确认手势表示已注意道路
            state.is_warning = False
            play_specific_audio(no_warning_wav)
            
            # 记录安全确认日志
            self.logger.log_interaction(
                self.current_user,
                "安全确认",
                f"通过{gesture}手势确认安全状态",
                "警告解除，恢复正常驾驶状态"
            )
            
            self.gesture_action.emit("safety_confirmed")
            print("通过手势确认安全状态")
        else:
            # 其他情况的确认
            self.logger.log_interaction(
                self.current_user,
                "手势交互",
                f"执行{gesture}确认手势",
                "确认操作完成"
            )
            
            self.gesture_action.emit("confirm")
    
    def handle_reject_gesture(self, gesture):
        """处理拒绝类手势（摇头、摇手）"""
        print(f"拒绝手势: {gesture}")
        
        if state.is_warning:
            # 在警告状态下拒绝确认
            self.logger.log_interaction(
                self.current_user,
                "安全警告",
                f"通过{gesture}手势拒绝警告确认",
                "驾驶员拒绝确认，警告状态保持"
            )
            
            self.gesture_action.emit("warning_rejected")
        else:
            # 其他情况的拒绝
            self.logger.log_interaction(
                self.current_user,
                "手势交互",
                f"执行{gesture}拒绝手势",
                "拒绝操作完成"
            )
            
            self.gesture_action.emit("reject")
    
    def handle_stop_gesture(self):
        """处理停止手势"""
        print("停止手势")
        
        # 记录停止手势日志
        self.logger.log_interaction(
            self.current_user,
            "手势控制",
            "执行停止手势",
            "停止当前操作"
        )
        
        self.gesture_action.emit("stop")
    
    def run(self):
        """主运行循环"""
        print("手势识别线程启动")
        
        # 记录线程运行开始日志
        self.logger.log_interaction(
            self.current_user,
            "系统状态",
            "手势识别主线程开始运行",
            "进入手势处理循环"
        )
        
        while self.running:
            self.msleep(100)
            
        # 记录线程运行结束日志
        self.logger.log_interaction(
            self.current_user,
            "系统状态",
            "手势识别主线程结束运行",
            "退出手势处理循环"
        )
        
        print("手势识别线程结束")