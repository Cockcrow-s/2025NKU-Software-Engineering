# face_thread.py
import sys
import os
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap

# 添加visual_interaction模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'visual_interaction'))

try:
    from inference import VisualRecognitionInterface
except ImportError as e:
    print(f"Warning: Cannot import VisualRecognitionInterface: {e}")
    VisualRecognitionInterface = None

import state


class FaceThread(QThread):
    # 定义信号
    frame_ready = pyqtSignal(np.ndarray)  # 发送帧数据
    distraction_detected = pyqtSignal(bool, int)  # 分心检测信号
    gesture_detected = pyqtSignal(str)  # 手势检测信号
    gaze_data = pyqtSignal(dict)  # 视线数据信号
    error_occurred = pyqtSignal(str)  # 错误信号
    
    def __init__(self, parent=None, camera_index=0):
        super().__init__(parent)
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.visual_interface = None
        self.frame_count = 0
    def init_camera(self):
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {self.camera_index}")
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"✓ 摄像头初始化成功 (索引: {self.camera_index})")
            return True
        except Exception as e:
            print(f"✗ 摄像头初始化失败: {str(e)}")
            self.error_occurred.emit(f"Camera initialization failed: {str(e)}")
            return False
    
    def init_visual_recognition(self):
        """初始化视觉识别模块"""
        if VisualRecognitionInterface is None:
            self.error_occurred.emit("Visual recognition module not available")
            return False
            
        try:
            # 获取权重文件路径
            weight_path = os.path.join(
                os.path.dirname(__file__), 
                '..', 
                'visual_interaction', 
                'weights', 
                '18.pt'
            )
            
            self.visual_interface = VisualRecognitionInterface(
                model_name="resnet18",
                weight_path=weight_path,
                dataset="mpiigaze"
            )            # 设置回调函数
            self.visual_interface.set_distraction_callback(self.on_distraction_detected)
            self.visual_interface.set_gesture_callback(self.on_gesture_detected)
            
            print("✓ 视觉识别系统初始化成功")
            print("  - 视线追踪模块已启动")
            print("  - 分心检测模块已启动") 
            print("  - 手势识别模块已启动")
            return True
        except Exception as e:
            self.error_occurred.emit(f"Visual recognition initialization failed: {str(e)}")
            return False
    
    def on_distraction_detected(self, distracted, warning_level):
        """分心检测回调"""
        state.is_warning = distracted
        self.distraction_detected.emit(distracted, warning_level)
        
    def on_gesture_detected(self, gesture):
        """手势检测回调"""
        self.gesture_detected.emit(gesture)
    def start_recognition(self):
        """开始识别"""
        print("🚀 启动车载智能交互系统...")
        if self.init_camera() and self.init_visual_recognition():
            self.running = True
            self.start()
            print("✓ 系统启动完成，开始监控...")
        else:
            print("✗ 系统启动失败")
            self.error_occurred.emit("Failed to initialize camera or visual recognition")
    
    def stop_recognition(self):
        """停止识别"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.quit()
        self.wait()
    
    def run(self):
        """主运行循环"""
        if not self.cap or not self.visual_interface:
            return
            
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            try:
                # 处理帧
                result = self.visual_interface.process_frame(frame)
                
                # 发送帧数据
                self.frame_ready.emit(frame)
                
                # 发送视线数据
                self.gaze_data.emit(result)
                
                self.frame_count += 1
                
                # 控制帧率
                self.msleep(33)  # ~30 FPS
                
            except Exception as e:
                self.error_occurred.emit(f"Frame processing error: {str(e)}")
                
        if self.cap:
            self.cap.release()
