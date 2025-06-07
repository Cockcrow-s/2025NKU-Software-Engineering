# gesture_thread.py
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import state


class GestureThread(QThread):
    # 定义信号
    gesture_recognized = pyqtSignal(str)  # 手势识别信号
    gesture_action = pyqtSignal(str)  # 手势动作信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.last_gesture = None
        self.gesture_count = {}
        
    def start_gesture_recognition(self):
        """开始手势识别"""
        self.running = True
        self.start()
    
    def stop_gesture_recognition(self):
        """停止手势识别"""
        self.running = False
        self.quit()
        self.wait()
    def process_gesture(self, gesture):
        """处理从face_thread接收到的手势"""
        if gesture and gesture != self.last_gesture:
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
    
    def handle_confirm_gesture(self, gesture):
        """处理确认类手势（点头、大拇指、OK手势）"""
        print(f"✅ 确认手势: {gesture}")
        if state.is_warning:
            # 如果当前在警告状态，确认手势表示已注意道路
            state.is_warning = False
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
        while self.running:
            # 手势线程主要通过信号接收数据，这里保持运行状态
            self.msleep(100)
