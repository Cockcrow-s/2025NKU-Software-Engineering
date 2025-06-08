"""
手势识别模块 - 检测诸如大拇指、OK、摇手等手势
集成到视觉识别接口中
"""
import cv2
import numpy as np
import time
import mediapipe as mp
from collections import deque

class HandGestureRecognizer:
    """手部手势识别器，支持大拇指、OK手势、摇手等手势"""
    
    def __init__(self):
        # 初始化MediaPipe手势识别模型
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 手势轨迹记录
        self.hand_positions = deque(maxlen=10)  # 记录最近10帧的手部位置
        
        # 上次检测到的手势
        self.last_gesture = None
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 0.5  # 手势冷却时间，避免频繁触发
        
    def detect_gesture(self, frame):
        """
        检测手势
        :param frame: 输入图像帧
        :return: 检测到的手势名称，如果没有检测到则返回None
        """
        # 转换到RGB颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 进行手势检测
        results = self.hands.process(frame_rgb)
        
        # 如果没有检测到手，返回None
        if not results.multi_hand_landmarks:
            return None
            
        # 获取第一只手
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # 记录手部位置（以食指指尖为参考点）
        h, w, _ = frame.shape
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
        self.hand_positions.append((x, y))
        
        # 提取手势特征
        gesture = self._recognize_gesture(hand_landmarks, w, h)
        
        # 检测波动手势（摇手）
        wave_gesture = self._detect_wave_gesture()
        if wave_gesture:
            gesture = wave_gesture
            
        # 应用手势冷却时间
        current_time = time.time()
        if gesture and (gesture != self.last_gesture or 
                       current_time - self.last_gesture_time > self.gesture_cooldown):
            self.last_gesture = gesture
            self.last_gesture_time = current_time
            return gesture
        
        return None
    
    def _recognize_gesture(self, hand_landmarks, frame_width, frame_height):
        """
        识别静态手势（大拇指、OK手势、停止手势）
        :param hand_landmarks: 手部关键点
        :param frame_width: 帧宽度
        :param frame_height: 帧高度
        :return: 识别到的手势名称
        """
        # 提取关键点坐标
        points = {}
        for point_id, landmark in enumerate(hand_landmarks.landmark):
            x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
            points[point_id] = (x, y)
            
        # 大拇指
        thumb_tip = points[4]
        thumb_ip = points[3]
        thumb_mcp = points[2]
        index_finger_tip = points[8]
        
        # 计算大拇指与其他手指的相对位置
        thumbs_up = self._is_thumbs_up(points)
        if thumbs_up:
            return "thumbs_up"
            
        # OK手势：大拇指和食指形成一个圆圈
        ok_gesture = self._is_ok_gesture(points)
        if ok_gesture:
            return "ok"
        
        # 停止手势：手掌张开，五指分开
        stop_gesture = self._is_stop_gesture(points)
        if stop_gesture:
            return "stop"
            
        return None
        
    def _is_thumbs_up(self, points):
        """检测大拇指向上手势"""
        # 大拇指关键点
        thumb_tip = points[4]
        thumb_ip = points[3]
        thumb_mcp = points[2]
        wrist = points[0]
        
        # 其他指尖
        fingers_tips = [points[8], points[12], points[16], points[20]]  # 食指、中指、无名指、小指指尖
        
        # 检查大拇指是否伸直且指向上方
        thumb_straight = (thumb_tip[1] < thumb_ip[1] and thumb_ip[1] < thumb_mcp[1])
        
        # 检查其他手指是否弯曲（指尖y坐标大于关节y坐标）
        fingers_bent = all(points[i][1] > points[i-2][1] for i in [8, 12, 16, 20])
        
        return thumb_straight and fingers_bent
    
    def _is_ok_gesture(self, points):
        """检测OK手势"""
        # 大拇指和食指指尖
        thumb_tip = points[4]
        index_finger_tip = points[8]
        
        # 检查大拇指和食指指尖是否接近
        distance = np.sqrt((thumb_tip[0] - index_finger_tip[0])**2 + 
                          (thumb_tip[1] - index_finger_tip[1])**2)
        
        # 接近阈值（可根据需要调整）
        threshold = 20  # 像素
        
        # 中指、无名指、小指应该伸直
        middle_up = points[12][1] < points[9][1]
        ring_up = points[16][1] < points[13][1]
        pinky_up = points[20][1] < points[17][1]
        
        return distance < threshold and middle_up and ring_up and pinky_up
    
    def _is_stop_gesture(self, points):
        """检测停止手势（手掌张开）"""
        # 所有指尖
        finger_tips = [points[4], points[8], points[12], points[16], points[20]]
        finger_mcp = [points[2], points[5], points[9], points[13], points[17]]  # 掌指关节
        
        # 检查所有手指是否伸直
        fingers_straight = all(tip[1] < mcp[1] for tip, mcp in zip(finger_tips, finger_mcp))
        
        # 检查手指是否分开（食指到小指）
        fingers_apart = True
        for i in range(1, 4):  # 相邻手指
            finger1_tip = finger_tips[i]
            finger2_tip = finger_tips[i+1]
            distance = np.sqrt((finger1_tip[0] - finger2_tip[0])**2 + 
                              (finger1_tip[1] - finger2_tip[1])**2)
            if distance < 30:  # 阈值可调
                fingers_apart = False
                break
                
        return fingers_straight and fingers_apart
    
    def _detect_wave_gesture(self):
        """检测摇手动作（基于手部轨迹）"""
        if len(self.hand_positions) < 5:
            return None
            
        # 计算最近几帧中手部位置的水平移动
        x_positions = [pos[0] for pos in self.hand_positions]
        
        # 计算水平方向变化
        deltas = [x_positions[i] - x_positions[i-1] for i in range(1, len(x_positions))]
        
        # 检测方向变化（左右摇摆至少两次）
        direction_changes = 0
        current_direction = None
        
        for delta in deltas:
            if abs(delta) < 5:  # 忽略很小的移动
                continue
                
            new_direction = "right" if delta > 0 else "left"
            
            if current_direction and new_direction != current_direction:
                direction_changes += 1
                
            current_direction = new_direction
        
        # 如果方向变化超过2次，且移动幅度大，判断为摇手
        significant_movement = max(x_positions) - min(x_positions) > 50
        
        if direction_changes >= 2 and significant_movement:
            return "wave"
            
        return None
        
    def draw_landmarks(self, frame, gesture=None):
        """
        在帧上绘制手部关键点和识别到的手势
        :param frame: 输入图像帧
        :param gesture: 识别到的手势
        :return: 绘制后的帧
        """
        # 转换到RGB颜色空间（MediaPipe需要RGB图像）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 进行手势检测
        results = self.hands.process(frame_rgb)
        
        # 如果检测到手，绘制关键点
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
            # 显示手势名称
            if gesture:
                gesture_text = {
                    "thumbs_up": "大拇指 (确认)",
                    "ok": "OK手势 (确认)",
                    "wave": "摇手 (拒绝)",
                    "stop": "停止手势"
                }.get(gesture, gesture)
                
                cv2.putText(frame, f"手势: {gesture_text}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
