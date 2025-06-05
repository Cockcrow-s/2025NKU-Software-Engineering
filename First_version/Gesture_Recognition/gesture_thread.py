# gesture_thread.py
"""
使用SVC分类器学习手势识别
"""

import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np
import joblib
from tqdm import tqdm
from PyQt5.QtCore import QThread, pyqtSignal
import sys
from PyQt5.QtWidgets import QApplication
import os

class GestureThread(QThread):
    # 初始化 MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils
    # 定义手势识别函数
    label_map = {
        0: "Unknown",
        1: "Fist",
        2: "Thumb Up",
        3: "Palm"
    }
    # 加载训练好的模型
    model = joblib.load('./model/gesture.joblib')
    # 检测到的手势信号
    gesture_detected = pyqtSignal(str)

    @staticmethod
    def min_max_normalize(data):
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

    @staticmethod
    def preprocess_landmarks(hand_landmarks):
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        data = np.array(landmarks)
        normalized = GestureThread.min_max_normalize(data)
        return normalized.reshape(1, -1)       
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.palm_count = 0
        self.last_palm_positions = []
        self.wave_threshold = 30
        self.required_palm_count = 3  # 需要检测的Palm次数

    def is_position_changed(self, current_pos):
        """检查手掌位置是否有足够变化"""
        if not self.last_palm_positions:
            return True
        last_pos = self.last_palm_positions[-1]
        distance = np.linalg.norm(np.array(current_pos) - np.array(last_pos))
        return distance > self.wave_threshold

    def run(self):
        cap = cv2.VideoCapture(0)
        self._is_running = True
        
        try:
            while self._is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )

                        input_data = self.preprocess_landmarks(hand_landmarks)
                        gesture_label = self.model.predict(input_data)[0]
                        gesture_name = self.label_map.get(gesture_label, "Unknown")

                        # 获取手腕位置用于挥手检测
                        wrist_pos = (
                            hand_landmarks.landmark[0].x, 
                            hand_landmarks.landmark[0].y
                        )

                        # 如果是Palm手势
                        if gesture_name == "Palm":
                            if self.is_position_changed(wrist_pos):
                                self.palm_count += 1
                                self.last_palm_positions.append(wrist_pos)
                                if len(self.last_palm_positions) > self.required_palm_count:
                                    self.last_palm_positions.pop(0)
                            
                            # 检测到足够次数的Palm
                            if self.palm_count >= self.required_palm_count:
                                self.gesture_detected.emit("wave_hand")
                                self._is_running = False
                                break
                        elif gesture_name != "Unknown":
                            # 检测到非Palm手势，立即退出并返回手势名称
                            self.gesture_detected.emit(gesture_name.lower().replace(" ", "_"))
                            self._is_running = False
                            break

                        # 显示信息
                        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Gesture Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.gesture_detected.emit("user_quit")  # 用户主动退出
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    # 必须使用 QApplication
    app = QApplication(sys.argv)
    
    try:
        # 检查模型是否存在
        if not os.path.exists('./model/gesture.joblib'):
            raise FileNotFoundError("模型文件不存在")
            
        thread = GestureThread()
        
        # 添加调试输出
        print("正在启动手势识别线程...")
        
        def handle_gesture(message):
            print(f"接收到手势信号: {message}")
            app.quit()
            
        thread.gesture_detected.connect(handle_gesture)
        thread.start()
        
        print("程序已启动，等待手势识别...")
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"程序启动失败: {e}")
        sys.exit(1)
