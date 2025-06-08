# gesture_thread.py
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import state
import cv2
from firstinteraction import play_specific_audio
import time
no_warning_wav = "CarSoft/no_warning.wav"

class GestureThread(QThread):
    # 定义信号
    gesture_recognized = pyqtSignal(str)  # 手势识别信号
    gesture_action = pyqtSignal(str)  # 手势动作信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.last_gesture = None
        self.gesture_count = {}
        # 添加手势识别所需的变量
        self.gesture_cooldown = 2.0  # 手势识别冷却时间，避免连续多次触发
        self.last_gesture_time = 0
        
    def start_gesture_recognition(self):
        """开始手势识别"""
        self.running = True
        self.start()
    
    def stop_gesture_recognition(self):
        """停止手势识别"""
        print("🛑 停止手势识别线程...")
        self.running = False
        self.quit()
        self.wait(3000)  # 等待最多3秒
        if self.isRunning():
            self.terminate()  # 强制终止
            self.wait(1000)
        print("✓ 手势识别线程已停止")
    
        
    def process_gesture(self, gesture):
        """处理从face_thread接收到的手势"""
        current_time = time.time()
        
        # 添加手势防抖处理和冷却时间
        if gesture and (gesture != self.last_gesture or 
                       current_time - self.last_gesture_time > self.gesture_cooldown):
            self.gesture_recognized.emit(gesture)
            print(f"🤏 检测到手势: {gesture}")
            
            # 处理手势动作
            if gesture in ['nod', 'thumbs_up', 'ok']:
                self.handle_confirm_gesture(gesture)
            elif gesture in ['shake', 'wave']:
                self.handle_reject_gesture(gesture)
            elif gesture == 'stop':
                self.handle_stop_gesture()
                
            self.last_gesture = gesture
            self.last_gesture_time = current_time
    
    def handle_confirm_gesture(self, gesture):
        """处理确认类手势（点头、大拇指、OK手势）"""
        print(f"✅ 确认手势: {gesture}")
        if state.is_warning:
            # 如果当前在警告状态，确认手势表示已注意道路
            state.is_warning = False
            play_specific_audio(no_warning_wav)
            self.gesture_action.emit("safety_confirmed")
            print("🛡️  通过手势确认安全状态")
        else:
            # 其他情况的确认
            self.gesture_action.emit("confirm")
    
    def handle_reject_gesture(self, gesture):
        """处理拒绝类手势（摇头、摇手）"""
        print(f"❌ 拒绝手势: {gesture}")
        if state.is_warning:
            # 在警告状态下拒绝确认
            self.gesture_action.emit("warning_rejected")
            print("⚠️  驾驶员拒绝警告确认")
        else:
            # 其他情况的拒绝
            self.gesture_action.emit("reject")
    
    def handle_stop_gesture(self):
        """处理停止手势"""
        print("✋ 停止手势")
        self.gesture_action.emit("stop")
    
    def run(self):
        """主运行循环"""
        print("🖐️ 手势识别线程启动")
        while self.running:
            self.msleep(100)
        print("🔚 手势识别线程结束")
