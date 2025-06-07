import cv2
import logging
import argparse
import warnings
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
import threading
from queue import Queue

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze

import uniface

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')


class VisualRecognitionInterface:
    """
    视觉识别接口类，提供给face_thread.py调用
    整合了视线追踪和头部姿势识别功能
    """
    def __init__(self, model_name="resnet18", weight_path="weights/18.pt", dataset="mpiigaze"):
        """
        初始化视觉识别接口
        :param model_name: 视线估计模型名称
        :param weight_path: 权重文件路径
        :param dataset: 数据集名称，用于获取配置
        """
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        
        # 根据数据集加载配置
        if dataset in data_config:
            self.config = data_config[dataset]
            self.bins = self.config["bins"]
            self.binwidth = self.config["binwidth"]
            self.angle = self.config["angle"]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # 创建索引张量
        self.idx_tensor = torch.arange(self.bins, device=self.device, dtype=torch.float32)
        
        # 初始化人脸检测器
        self.face_detector = uniface.RetinaFace()
          # 加载视线估计模型
        print("🔍 正在加载视线估计模型...")
        try:
            self.gaze_detector = get_model(model_name, self.bins, inference_mode=True)
            print(f"✓ 模型架构加载成功：{model_name}")
            
            print(f"🔧 正在加载模型权重：{weight_path}")
            state_dict = torch.load(weight_path, map_location=self.device)
            self.gaze_detector.load_state_dict(state_dict)
            self.gaze_detector.to(self.device)
            self.gaze_detector.eval()
            
            print(f"✅ 视线估计模型加载成功！")
            print(f"   - 设备：{self.device}")
            print(f"   - 数据集：{dataset}")
            print(f"   - 角度分bins：{self.bins}")
            logging.info(f"✅ Gaze Estimation model '{model_name}' weights loaded successfully on {self.device}.")
        except Exception as e:
            print(f"❌ 模型加载失败：{e}")
            logging.error(f"❌ Exception occurred while loading gaze model: {e}")
            raise
          # 头部姿势识别器
        self.head_gesture_recognizer = HeadGestureRecognizer()
        
        # 分心检测相关变量
        self.safe_angle = 0.1  # 设置为固定的安全角度值（0.1弧度，约5.7度）
        self.distraction_start_time = None
        self.distracted = False
        self.warning_level = 0
        self.distraction_duration = 0
        # 回调函数
        self.distraction_callback = None
        self.gesture_callback = None
        
        # 视线历史
        self.angle_history = deque(maxlen=100)
    
    def set_distraction_callback(self, callback):
        """设置分心检测回调函数"""
        self.distraction_callback = callback
        
    def set_gesture_callback(self, callback):
        """设置手势检测回调函数"""
        self.gesture_callback = callback
    
    def process_frame(self, frame):
        """
        处理单帧图像
        :param frame: 输入帧
        :return: 包含处理结果的字典
        """
        result = {
            "distracted": False, 
            "warning_level": 0,
            "gaze_angle": 0,
            "gesture": None,
            "face_detected": False,
            "face_info": None
        }
        
        if frame is None:
            return result
        
        # 人脸检测
        bboxes, keypoints = self.face_detector.detect(frame)
        
        # 处理头部姿势
        head_gesture = None
        if len(bboxes) > 0:
            # 取第一个检测到的人脸
            keypoints_first_face = keypoints[0]
            
            # 估计头部姿势
            pitch, yaw, roll = self.head_gesture_recognizer.estimate_head_pose(keypoints_first_face, frame.shape)
            
            if pitch is not None and yaw is not None:
                # 检测头部姿势
                head_gesture = self.head_gesture_recognizer.detect_gesture(pitch, yaw, roll)
                
                # 如果有手势，调用回调
                if head_gesture and self.gesture_callback:
                    self.gesture_callback(head_gesture)
                
                # 更新结果
                result["gesture"] = head_gesture
            
            # 更新结果中的人脸信息
            result["face_detected"] = True
            bbox = bboxes[0]
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            h, w = frame.shape[:2]
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            
            # 跳过无效框
            if x_max <= x_min or y_max <= y_min:
                return result
                
            face_image = frame[y_min:y_max, x_min:x_max]
            
            # 跳过空图像
            if face_image.size == 0:
                return result
            
            # 视线估计处理
            try:
                image = pre_process(face_image)
                image = image.to(self.device)
                
                with torch.no_grad():
                    pitch, yaw = self.gaze_detector(image)
                    pitch_predicted, yaw_predicted = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)
                    pitch_predicted = torch.sum(pitch_predicted * self.idx_tensor, dim=1) * self.binwidth - self.angle
                    yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor, dim=1) * self.binwidth - self.angle
                    pitch_predicted = np.radians(pitch_predicted.cpu().numpy()[0])
                    yaw_predicted = np.radians(yaw_predicted.cpu().numpy()[0])
                  # 计算视线与正前方的3D夹角theta
                theta = np.arccos(np.cos(pitch_predicted) * np.cos(yaw_predicted))
                self.angle_history.append(theta)
                
                # 分心检测逻辑 - 驾驶场景
                current_time = time.time()
                if abs(theta) > self.safe_angle:
                    if self.distraction_start_time is None:
                        self.distraction_start_time = current_time  # 记录分心起始时间
                    else:
                        self.distraction_duration = current_time - self.distraction_start_time
                        
                        # 按照需求，超过3秒认为分心
                        if self.distraction_duration > 3:
                            self.distracted = True
                            
                            # 动态调整警告等级：
                            # 1级: 3-5秒 轻度警告
                            # 2级: 5-8秒 中度警告
                            # 3级: >8秒 严重警告
                            if self.distraction_duration > 8:
                                self.warning_level = 3  # 严重警告
                            elif self.distraction_duration > 5:
                                self.warning_level = 2  # 中度警告
                            else:
                                self.warning_level = 1  # 轻度警告
                else:
                    # 视线回到前方，但保持短暂的缓冲期
                    if self.distracted and self.distraction_start_time is not None:
                        # 如果之前处于分心状态，给予1秒的缓冲期
                        buffer_time = 1.0
                        if current_time - self.distraction_start_time < buffer_time:
                            # 保持分心状态，但不增加警告等级
                            pass
                        else:
                            self.distraction_start_time = None
                            self.distracted = False
                            self.warning_level = 0
                            self.distraction_duration = 0
                    else:
                        self.distraction_start_time = None
                        self.distracted = False
                        self.warning_level = 0
                        self.distraction_duration = 0
                
                # 调用分心回调
                if self.distraction_callback and (self.distracted or result["distracted"] != self.distracted):
                    self.distraction_callback(self.distracted, self.warning_level)
                
                # 更新结果
                result["distracted"] = self.distracted
                result["warning_level"] = self.warning_level
                result["gaze_angle"] = theta
                result["face_info"] = {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "keypoints": keypoints_first_face,
                    "pitch": pitch_predicted,
                    "yaw": yaw_predicted,
                    "head_pitch": pitch,
                    "head_yaw": yaw,
                    "head_roll": roll
                }
                
            except Exception as e:
                logging.error(f"Error in gaze estimation: {e}")
        
        return result
        
    def render_info_on_frame(self, frame, result):
        """
        在帧上渲染分析信息
        :param frame: 输入帧
        :param result: process_frame返回的结果
        :return: 渲染后的帧
        """
        if not result["face_detected"]:
            return frame
            
        face_info = result["face_info"]
        bbox = face_info["bbox"]
        keypoints = face_info["keypoints"]
        
        # 绘制人脸边界框
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
          # 绘制关键点
        for i, kp in enumerate(keypoints):
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            # 标注关键点
            labels = ["L_Eye", "R_Eye", "Nose", "L_Mouth", "R_Mouth"]
            if i < len(labels):
                cv2.putText(frame, labels[i], (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        # 绘制视线方向
        draw_bbox_gaze(frame, bbox, face_info["pitch"], face_info["yaw"])
        
        # 绘制视线角度信息
        theta = result['gaze_angle']
        safe_threshold = 0.1  # 安全角度阈值
        
        # 根据视线角度设置颜色
        if theta > safe_threshold:
            color = (0, 0, 255)  # 红色 - 分心
            status = "DISTRACTED"
        else:
            color = (0, 255, 0)  # 绿色 - 正常
            status = "FOCUSED"
            
        # 显示视线状态
        cv2.putText(frame, f"Gaze: {status}", 
                   (x_min, y_min-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Theta: {theta:+.3f} rad", 
                    (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
        # 绘制视线追踪历史轨迹
        if hasattr(self, 'angle_history') and len(self.angle_history) > 1:
            history_points = list(self.angle_history)
            for i in range(1, len(history_points)):
                # 将角度值映射到屏幕坐标
                y1 = int(30 + history_points[i-1] * 200)  # 放大显示
                y2 = int(30 + history_points[i] * 200)
                x1 = frame.shape[1] - len(history_points) + i - 1
                x2 = frame.shape[1] - len(history_points) + i
                
                # 根据角度值设置轨迹颜色
                if history_points[i] > safe_threshold:
                    trail_color = (0, 0, 255)  # 红色轨迹
                else:
                    trail_color = (0, 255, 0)  # 绿色轨迹
                    
                cv2.line(frame, (x1, y1), (x2, y2), trail_color, 2)
            
            # 绘制安全线
            safe_y = int(30 + safe_threshold * 200)
            cv2.line(frame, (frame.shape[1]-100, safe_y), 
                    (frame.shape[1]-10, safe_y), (255, 255, 0), 2)
            cv2.putText(frame, "Safe Line", 
                       (frame.shape[1]-100, safe_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # 显示头部姿势
        if result["gesture"] == "nod":
            cv2.putText(frame, "Nod (Confirm)", (30, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        elif result["gesture"] == "shake":
            cv2.putText(frame, "Shake (Reject)", (30, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)        # 分心警告显示 - 增强的红色边框警告
        if result["distracted"]:
            now = time.time()
            # 根据警告等级调整闪烁频率
            flash_frequency = 2 + result["warning_level"]  # 每秒闪烁次数
            flash = int(now * flash_frequency) % 2 == 0
            
            # 动态调整警告文本和颜色
            warning_level = result["warning_level"]
            if warning_level == 1:
                warning_text = "⚠️ 警告！请目视前方"
                border_color = (0, 100, 255)  # 橙红色
            elif warning_level == 2:
                warning_text = "🚨 警告！请立即目视前方"
                border_color = (0, 50, 255)   # 深红色
            else:  # 级别3
                warning_text = "🆘 危险！请立即目视前方！"
                border_color = (0, 0, 255)    # 纯红色
            
            if flash:
                # 绘制多层红色边框警告
                h, w = frame.shape[:2]
                
                # 外层边框 - 最粗
                border_thickness = 8 + warning_level * 4
                cv2.rectangle(frame, (0, 0), (w-1, h-1), border_color, border_thickness)
                
                # 中层边框
                cv2.rectangle(frame, (15, 15), (w-16, h-16), (0, 0, 255), 6)
                
                # 内层边框
                cv2.rectangle(frame, (25, 25), (w-26, h-26), (255, 255, 255), 2)
                
                # 在四个角落添加警告标志
                corner_size = 30
                # 左上角
                cv2.rectangle(frame, (0, 0), (corner_size, corner_size), (0, 0, 255), -1)
                cv2.putText(frame, "!", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 右上角
                cv2.rectangle(frame, (w-corner_size, 0), (w, corner_size), (0, 0, 255), -1)
                cv2.putText(frame, "!", (w-22, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 左下角
                cv2.rectangle(frame, (0, h-corner_size), (corner_size, h), (0, 0, 255), -1)
                cv2.putText(frame, "!", (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 右下角
                cv2.rectangle(frame, (w-corner_size, h-corner_size), (w, h), (0, 0, 255), -1)
                cv2.putText(frame, "!", (w-22, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 显示警告文本 - 居中显示
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = 100
            
            # 文本背景
            cv2.rectangle(frame, (text_x-10, text_y-35), 
                         (text_x+text_size[0]+10, text_y+10), (0, 0, 0), -1)
            cv2.rectangle(frame, (text_x-10, text_y-35), 
                         (text_x+text_size[0]+10, text_y+10), (0, 0, 255), 3)
            
            # 警告文本
            cv2.putText(frame, warning_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # 显示分心持续时间
            if hasattr(self, 'distraction_duration'):
                duration_text = f"分心时长: {self.distraction_duration:.1f}s"
                cv2.putText(frame, duration_text, (text_x, text_y+30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # 根据警告级别调整边框粗细
                border_thickness = 4 + warning_level * 2
                cv2.rectangle(frame, (20, 20), 
                             (frame.shape[1]-20, frame.shape[0]-20), 
                             (0,0,255), border_thickness)  # 红色闪烁边框
                
                # 高级别警告额外视觉效果
                if warning_level >= 3:
                    # 添加额外的警告指示
                    cv2.rectangle(frame, (40, 40), 
                                 (frame.shape[1]-40, frame.shape[0]-40), 
                                 (0,255,255), border_thickness-2)  # 黄色内边框
        
        return frame


class HeadGestureRecognizer:
    def __init__(self, frame_queue=None):
        # 使用RetinaFace替代MediaPipe
        self.face_detector = uniface.RetinaFace()
        self.last_pitch = None  # 上一帧pitch
        self.last_yaw = None    # 上一帧yaw
        self.last_roll = None   # 上一帧roll
        self.nod_counter = 0    # 点头计数
        self.shake_counter = 0  # 摇头计数
        self.frame_queue = frame_queue  # 主进程传入的帧队列
        self.nod_threshold = 0.05  # 点头判据阈值
        self.shake_threshold = 0.05  # 摇头判据阈值
        self.roll_limit = 0.3  # 侧倾过大时不判定
        self.nod_frames = 3    # 点头需连续帧
        self.shake_frames = 3  # 摇头需连续帧
        # 上一帧关键点位置
        self.prev_keypoints = None

    def estimate_head_pose(self, keypoints, image_shape):
        """
        基于人脸关键点简单估计头部姿态（pitch/yaw/roll）
        :param keypoints: RetinaFace检测的人脸5个关键点
        :param image_shape: 图像尺寸
        :return: pitch, yaw, roll
        """
        # 注意：此为简化实现，实际效果可能不如mediapipe的精确
        image_h, image_w = image_shape[:2]
        
        # 提取眼睛、鼻尖和嘴巴关键点
        left_eye = keypoints[0]
        right_eye = keypoints[1]
        nose = keypoints[2]
        left_mouth = keypoints[3]
        right_mouth = keypoints[4]
        
        # 计算眼睛中心点和嘴巴中心点
        eye_center = [(left_eye[0] + right_eye[0])/2, (left_eye[1] + right_eye[1])/2]
        mouth_center = [(left_mouth[0] + right_mouth[0])/2, (left_mouth[1] + right_mouth[1])/2]
        
        # 计算眼睛水平线斜率 (yaw)
        eye_dx = right_eye[0] - left_eye[0]
        eye_dy = right_eye[1] - left_eye[1]
        if eye_dx != 0:
            yaw = np.arctan(eye_dy / eye_dx)  # 左右转动
        else:
            yaw = 0
            
        # 计算鼻子到眼睛中心和嘴巴中心的垂直关系 (pitch)
        vertical_line = [eye_center[1], mouth_center[1]]
        nose_offset = nose[1] - (eye_center[1] + mouth_center[1])/2
        pitch = nose_offset / (vertical_line[1] - vertical_line[0]) * 0.5  # 上下点头
        
        # 计算眼睛线相对水平线的旋转 (roll)
        eye_angle = np.arctan2(eye_dy, eye_dx)
        roll = eye_angle  # 头部倾斜
        
        return pitch, yaw, roll

    def detect_gesture(self, pitch, yaw, roll):
        """
        检测点头/摇头动作：
        - 点头：pitch上下一个来回且幅度都超过阈值
        - 摇头：yaw左右一个来回且幅度都超过阈值
        :return: 'nod'/'shake'/None
        """
        gesture = None
        # 点头检测
        if self.last_pitch is not None:
            delta_pitch = pitch - self.last_pitch
            # pitch方向变化状态机
            if not hasattr(self, 'pitch_state'):
                self.pitch_state = 0  # 0:初始, 1:向上, 2:向下
            if self.pitch_state == 0 and delta_pitch > self.nod_threshold:
                self.pitch_state = 1
                self.pitch_peak = pitch
            elif self.pitch_state == 1 and delta_pitch < -self.nod_threshold:
                if abs(self.pitch_peak - pitch) > self.nod_threshold * 2:
                    gesture = 'nod'
                self.pitch_state = 0
            elif self.pitch_state == 0 and delta_pitch < -self.nod_threshold:
                self.pitch_state = 2
                self.pitch_peak = pitch
            elif self.pitch_state == 2 and delta_pitch > self.nod_threshold:
                if abs(self.pitch_peak - pitch) > self.nod_threshold * 2:
                    gesture = 'nod'
                self.pitch_state = 0
        self.last_pitch = pitch
        
        # 摇头检测
        if self.last_yaw is not None:
            delta_yaw = yaw - self.last_yaw
            if not hasattr(self, 'yaw_state'):
                self.yaw_state = 0  # 0:初始, 1:向左, 2:向右
            if self.yaw_state == 0 and delta_yaw > self.shake_threshold:
                self.yaw_state = 1
                self.yaw_peak = yaw
            elif self.yaw_state == 1 and delta_yaw < -self.shake_threshold:
                if abs(self.yaw_peak - yaw) > self.shake_threshold * 2:
                    gesture = 'shake'
                self.yaw_state = 0
            elif self.yaw_state == 0 and delta_yaw < -self.shake_threshold:
                self.yaw_state = 2
                self.yaw_peak = yaw
            elif self.yaw_state == 2 and delta_yaw > self.shake_threshold:
                if abs(self.yaw_peak - yaw) > self.shake_threshold * 2:
                    gesture = 'shake'
                self.yaw_state = 0
        self.last_yaw = yaw
        
        return gesture

    def run(self):
        """
        从帧队列读取图像，检测头部姿态并可视化点头/摇头动作。
        """
        while True:
            if self.frame_queue is not None:
                frame = self.frame_queue.get()
                if frame is None:
                    break
            else:
                break
            
            # 使用RetinaFace进行人脸检测
            bboxes, keypoints = self.face_detector.detect(frame)
            gesture = None
            
            if len(bboxes) > 0:
                # 取第一个检测到的人脸
                keypoints_first_face = keypoints[0]
                
                # 估计头部姿势
                pitch, yaw, roll = self.estimate_head_pose(keypoints_first_face, frame.shape)
                
                if pitch is not None and yaw is not None:
                    gesture = self.detect_gesture(pitch, yaw, roll)
                    # 显示姿态角度
                    cv2.putText(frame, f"Pitch: {pitch:.2f} Yaw: {yaw:.2f} Roll: {roll:.2f}", 
                               (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    # 可视化点头/摇头
                    if gesture == 'nod':
                        cv2.putText(frame, "Nod (Confirm)", (30, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    elif gesture == 'shake':
                        cv2.putText(frame, "Shake (Reject)", (30, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    
                # 绘制人脸框和关键点
                bbox = bboxes[0]
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # 绘制关键点
                for kp in keypoints_first_face:
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                    
            cv2.imshow('Head Gesture Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()


def run_head_gesture(frame_queue, result_queue):
    """
    子线程：从帧队列读取图像，检测头部姿势并返回结果
    :param frame_queue: 输入帧队列
    :param result_queue: 输出结果队列
    """
    recognizer = HeadGestureRecognizer(frame_queue=frame_queue)
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
            
        # 使用RetinaFace进行人脸检测
        bboxes, keypoints = recognizer.face_detector.detect(frame)
        gesture = None
        
        if len(bboxes) > 0:
            # 取第一个检测到的人脸
            keypoints_first_face = keypoints[0]
            
            # 估计头部姿势
            pitch, yaw, roll = recognizer.estimate_head_pose(keypoints_first_face, frame.shape)
            
            if pitch is not None and yaw is not None:
                gesture = recognizer.detect_gesture(pitch, yaw, roll)
                # 绘制点头/摇头轨迹
                cx, cy = frame.shape[1]//2, frame.shape[0]//2
                if gesture == 'nod':
                    cv2.arrowedLine(frame, (cx, cy-60), (cx, cy+60), (0,0,255), 6, tipLength=0.3)
                    cv2.putText(frame, "Nod (Confirm)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                elif gesture == 'shake':
                    cv2.arrowedLine(frame, (cx-60, cy), (cx+60, cy), (255,0,0), 6, tipLength=0.3)
                    cv2.putText(frame, "Shake (Reject)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(frame, f"Pitch: {pitch:.2f} Yaw: {yaw:.2f} Roll: {roll:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
            # 绘制人脸框和关键点
            bbox = bboxes[0]
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # 绘制关键点
            for kp in keypoints_first_face:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                
        cv2.imshow('Head Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        result_queue.put(gesture)
    cv2.destroyAllWindows()


def parse_args():
    """
    解析命令行参数，包括模型类型、权重路径、输入源、输出路径、数据集等。
    根据数据集自动补充bins、binwidth、angle等参数。
    """
    parser = argparse.ArgumentParser(description="Gaze estimation inference")
    parser.add_argument("--model", type=str, default="resnet18", help="Model name, default `resnet18`")
    parser.add_argument(
        "--weight",
        type=str,
        default="weights/18.pt",
        help="Path to gaze esimation model weights"
    )
    parser.add_argument("--view", action="store_true", default=True, help="Display the inference results")
    parser.add_argument("--source", type=str, default="0",
                        help="Path to source video file or camera index")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output file")
    parser.add_argument("--dataset", type=str, default="mpiigaze", help="Dataset name to get dataset related configs")
    parser.add_argument("--with-head-gesture", action="store_true", help="Enable head gesture recognition")
    parser.add_argument("--demo-mode", action="store_true", help="Run in multi-task demo mode only")
    args = parser.parse_args()

    # Override default values based on selected dataset
    if args.dataset in data_config:
        dataset_config = data_config[args.dataset]
        args.bins = dataset_config["bins"]
        args.binwidth = dataset_config["binwidth"]
        args.angle = dataset_config["angle"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available options: {list(data_config.keys())}")

    return args


def pre_process(image):
    """
    对输入的人脸图像进行预处理：
    1. 转为RGB格式
    2. 缩放到448x448
    3. 转为Tensor并归一化
    返回：shape为(1,3,448,448)的Tensor
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch


def main(params):
    """
    推理主函数：
    1. 初始化模型和人脸检测器
    2. 进入主循环，逐帧检测人脸、估计视线、分心检测
    3. 实时绘制人脸框、视线向量、分心警告和yaw波形图
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    face_detector = uniface.RetinaFace()  # 第三方人脸检测库

    try:
        gaze_detector = get_model(params.model, params.bins, inference_mode=True)
        state_dict = torch.load(params.weight, map_location=device)
        gaze_detector.load_state_dict(state_dict)
        logging.info("Gaze Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading pre-trained weights of gaze estimation model. Exception: {e}")

    gaze_detector.to(device)    
    gaze_detector.eval()

    video_source = params.source
    if video_source.isdigit() or video_source == '0':
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)
        
    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(params.output, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
        
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    safe_angle = 0.1  # 设置为固定的安全角度值（0.1弧度，约5.7度）
    distraction_start_time = None
    distracted = False
    warning_level = 0
    last_warning_flash = 0

    # 波形图相关
    angle_history = deque(maxlen=100)
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_ylim(0, 0.5)  # 适合theta范围
    ax.set_title('Theta Deviation (rad)')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Theta (rad)')
    line, = ax.plot([], [], 'b-')

    # 头部姿势识别相关变量
    head_gesture_enabled = params.with_head_gesture
    head_gesture_text = ''
    frame_queue = None
    result_queue = None
    gesture_thread = None
    
    # 如果启用头部姿势识别，启动子线程
    if head_gesture_enabled:
        frame_queue = Queue(maxsize=5)
        result_queue = Queue(maxsize=5)
        gesture_thread = threading.Thread(target=run_head_gesture, args=(frame_queue, result_queue))
        gesture_thread.start()

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                logging.info("Failed to obtain frame or EOF")
                break
                
            # 如果启用头部姿势检测，将帧发送给子线程
            if head_gesture_enabled and frame_queue and not frame_queue.full():
                frame_queue.put(frame.copy())
                
                # 获取子线程识别结果
                while not result_queue.empty():
                    gesture = result_queue.get()
                    if gesture == 'nod':
                        head_gesture_text = 'Nod (Confirm)'
                    elif gesture == 'shake':
                        head_gesture_text = 'Shake (Reject)'
                    elif gesture is None:
                        head_gesture_text = ''
                        
                # 在主画面显示头部姿态识别结果
                if head_gesture_text:
                    cv2.putText(frame, head_gesture_text, (30, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                (0,0,255) if head_gesture_text.startswith('Nod') else (255,0,0), 2)

            bboxes, keypoints = face_detector.detect(frame)
            for bbox, keypoint in zip(bboxes, keypoints):
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
                h, w = frame.shape[:2]
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)
                if x_max <= x_min or y_max <= y_min:
                    continue  # 跳过无效框
                image = frame[y_min:y_max, x_min:x_max]
                if image.size == 0:
                    continue  # 跳过空图像
                image = pre_process(image)
                image = image.to(device)
                pitch, yaw = gaze_detector(image)
                pitch_predicted, yaw_predicted = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                pitch_predicted = np.radians(pitch_predicted.cpu().numpy()[0])
                yaw_predicted = np.radians(yaw_predicted.cpu().numpy()[0])
                draw_bbox_gaze(frame, bbox, pitch_predicted, yaw_predicted)
                # 始终绘制人脸检测框
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)                # 计算视线与正前方的3D夹角theta
                theta = np.arccos(np.cos(pitch_predicted) * np.cos(yaw_predicted))
                # 实时显示theta与正前方夹角
                cv2.putText(frame, f"Theta deviation: {theta:+.2f} rad", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                # 记录theta角度到波形历史
                angle_history.append(theta)
                # 分心检测逻辑
                if abs(theta) > safe_angle:
                    if distraction_start_time is None:
                        distraction_start_time = time.time()  # 记录分心起始时间
                    else:
                        duration = time.time() - distraction_start_time
                        if duration > 2:
                            distracted = True
                            warning_level = 1
                else:
                    distraction_start_time = None
                    distracted = False
                    warning_level = 0
                # 分心警告显示
                if distracted and warning_level > 0:
                    now = time.time()
                    flash = int(now*2)%2 == 0
                    if flash:
                        cv2.putText(frame, "WARNING: Distracted! Look Forward!", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 4)
                        cv2.rectangle(frame, (20, 20), (frame.shape[1]-20, frame.shape[0]-20), (0,0,255), 8)  # 红色闪烁边框

            if params.output:
                out.write(frame)
            if params.view:
                cv2.imshow('Demo', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # --- 波形图实时刷新 ---
            # 更新matplotlib波形窗口，显示最近100帧theta角度变化
            line.set_data(range(len(angle_history)), list(angle_history))
            ax.set_xlim(max(0, len(angle_history)-100), len(angle_history))
            fig.canvas.draw()
            fig.canvas.flush_events()    # 清理资源
    cap.release()
    if params.output:
        out.release()
    cv2.destroyAllWindows()
    
    # 结束头部姿势检测线程
    if head_gesture_enabled and frame_queue:
        frame_queue.put(None)  # 发送退出信号
        if gesture_thread:
            gesture_thread.join()


def run_multi_task_demo():
    """
    运行多任务演示，包括头部姿势识别和主摄像头显示
    """
    cap = cv2.VideoCapture(0)
    frame_queue = Queue(maxsize=5)
    result_queue = Queue(maxsize=5)
    gesture_thread = threading.Thread(target=run_head_gesture, args=(frame_queue, result_queue))
    gesture_thread.start()
    gesture_text = ''
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame.copy())
        # 获取子线程识别结果
        while not result_queue.empty():
            gesture = result_queue.get()
            if gesture == 'nod':
                gesture_text = 'Nod (Confirm)'
            elif gesture == 'shake':
                gesture_text = 'Shake (Reject)'
            elif gesture is None:
                gesture_text = ''
        if gesture_text:
            cv2.putText(frame, gesture_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if gesture_text.startswith('Nod') else (255,0,0), 2)
        cv2.imshow('Main Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frame_queue.put(None)
    gesture_thread.join()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()

    if args.demo_mode:
        # 仅运行多任务演示模式
        run_multi_task_demo()
    else:
        # 运行标准模式
        if not args.view and not args.output:
            raise Exception("At least one of --view or --ouput must be provided.")
        main(args)
