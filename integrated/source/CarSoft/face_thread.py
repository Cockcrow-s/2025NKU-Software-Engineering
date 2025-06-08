import sys
import os
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap

# 确保能找到所需模块
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..', 'visual_interaction'))

# 导入state模块
try:
    import state
except ImportError:
    print("⚠️  创建临时state模块")
    class TempState:
        is_warning = False
        is_playing = False
    state = TempState()

# 智能查找权重文件并决定使用哪种接口
def find_weight_file():
    """查找权重文件"""
    weight_paths = [
        # 当前目录下的weights文件夹
        os.path.join(current_dir, 'weights', '18.pt'),
        # 上级目录的visual_interaction/weights
        os.path.join(current_dir, '..', 'visual_interaction', 'weights', '18.pt'),
        # 绝对路径
        r"C:\Users\23301\Desktop\source\source\CarSoft\weights\18.pt"
    ]
    
    for path in weight_paths:
        if os.path.exists(path):
            print(f"✓ 找到权重文件: {path}")
            return path
    
    print("⚠️  未找到权重文件，搜索路径:")
    for path in weight_paths:
        print(f"   - {path}")
    return None

# 尝试导入视觉识别接口
VISUAL_RECOGNITION_AVAILABLE = False
VisualRecognitionInterface = None
WEIGHT_FILE_PATH = find_weight_file()

try:
    if WEIGHT_FILE_PATH:
        # 权重文件存在，尝试导入完整接口
        import torch
        from inference import VisualRecognitionInterface
        VISUAL_RECOGNITION_AVAILABLE = True
        print("✓ 完整视觉识别接口可用")
    else:
        # 权重文件不存在，使用简化接口
        raise ImportError("Weight file not found")
        
except ImportError as e:
    print(f"⚠️  完整接口不可用: {e}")
    print("✓ 使用简化视觉识别接口...")
    
    # 创建简化的视觉识别接口
    class SimpleVisualInterface:
        def __init__(self, model_name="opencv", weight_path=None, dataset="basic"):
            self.model_name = model_name
            self.weight_path = weight_path
            self.dataset = dataset
            self.distraction_callback = None
            self.gesture_callback = None
            
            # 初始化OpenCV检测器
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                print("✓ OpenCV人脸/眼部检测器初始化成功")
                self.detection_available = True
            except Exception as e:
                print(f"⚠️  OpenCV检测器初始化失败: {e}")
                self.detection_available = False
            
            # 分心检测相关变量
            self.face_history = []
            self.no_face_count = 0
            self.distraction_count = 0
            
            print("✓ 简化视觉识别接口初始化完成")
        
        def set_distraction_callback(self, callback):
            """设置分心检测回调"""
            self.distraction_callback = callback
            print("✓ 分心检测回调已设置")
        
        def set_gesture_callback(self, callback):
            """设置手势识别回调"""
            self.gesture_callback = callback
            print("✓ 手势识别回调已设置")
        
        def process_frame(self, frame):
            """处理视频帧"""
            result = {
                "status": "opencv_mode",
                "face_detected": False,
                "distracted": False,
                "warning_level": 0,
                "confidence": 0.0,
                "face_count": 0,
                "gaze_info": "OpenCV基础模式",
                "timestamp": cv2.getTickCount()
            }
            
            if not self.detection_available or frame is None:
                return result
            
            try:
                # 转换为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 人脸检测
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5,
                    minSize=(50, 50)
                )
                
                if len(faces) > 0:
                    result["face_detected"] = True
                    result["face_count"] = len(faces)
                    result["confidence"] = 0.8
                    
                    # 重置无脸计数
                    self.no_face_count = 0
                    
                    # 取最大的人脸进行分析
                    largest_face = max(faces, key=lambda face: face[2] * face[3])
                    (x, y, w, h) = largest_face
                    
                    # 计算人脸中心位置
                    face_center_x = x + w // 2
                    face_center_y = y + h // 2
                    frame_center_x = frame.shape[1] // 2
                    frame_center_y = frame.shape[0] // 2
                    
                    # 计算偏移程度
                    deviation_x = abs(face_center_x - frame_center_x) / frame_center_x
                    deviation_y = abs(face_center_y - frame_center_y) / frame_center_y
                    
                    # 分心检测逻辑
                    is_distracted = deviation_x > 0.25 or deviation_y > 0.2
                    
                    if is_distracted:
                        self.distraction_count += 1
                    else:
                        self.distraction_count = max(0, self.distraction_count - 1)
                    
                    # 根据连续分心帧数确定警告级别
                    if self.distraction_count > 15:
                        result["distracted"] = True
                        result["warning_level"] = 3
                    elif self.distraction_count > 10:
                        result["distracted"] = True
                        result["warning_level"] = 2
                    elif self.distraction_count > 5:
                        result["distracted"] = True
                        result["warning_level"] = 1
                    
                    # 调用分心检测回调
                    if self.distraction_callback:
                        self.distraction_callback(result["distracted"], result["warning_level"])
                    
                    result["gaze_info"] = f"人脸位置: ({face_center_x}, {face_center_y}), 偏移: {deviation_x:.2f}"
                
                else:
                    # 没有检测到人脸
                    self.no_face_count += 1
                    
                    if self.no_face_count > 30:
                        result["distracted"] = True
                        result["warning_level"] = 2
                        if self.distraction_callback:
                            self.distraction_callback(True, 2)
                        result["gaze_info"] = "未检测到人脸 - 可能离开座位"
                    else:
                        result["gaze_info"] = f"未检测到人脸 ({self.no_face_count}/30)"
                
            except Exception as e:
                print(f"简化处理错误: {e}")
                result["gaze_info"] = f"处理错误: {str(e)}"
            
            return result
    
    VisualRecognitionInterface = SimpleVisualInterface
    VISUAL_RECOGNITION_AVAILABLE = True
    print("✓ 简化视觉识别接口创建成功")


class FaceThread(QThread):
    # 定义信号
    frame_ready = pyqtSignal(np.ndarray)
    distraction_detected = pyqtSignal(bool, int)
    gesture_detected = pyqtSignal(str)
    gaze_data = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None, camera_index=0):
        super().__init__(parent)
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.visual_interface = None
        self.frame_count = 0
        self.camera_available = False
        self.visual_recognition_available = VISUAL_RECOGNITION_AVAILABLE

    def init_camera(self):
        """初始化摄像头"""
        try:
            print("🔍 检测可用摄像头...")
            # 尝试多个摄像头索引
            for index in [0, 1, 2]:
                print(f"   尝试摄像头索引 {index}...")
                self.cap = cv2.VideoCapture(index)
                if self.cap.isOpened():
                    # 测试读取一帧
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.camera_index = index
                        self.camera_available = True
                        print(f"✓ 摄像头初始化成功 (索引: {index})")
                        print(f"   分辨率: {frame.shape[1]}x{frame.shape[0]}")
                        
                        # 设置摄像头参数
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        return True
                    else:
                        print(f"   摄像头 {index} 无法读取帧")
                        self.cap.release()
                else:
                    print(f"   摄像头 {index} 无法打开")
                    if self.cap:
                        self.cap.release()
            
            # 如果所有摄像头都无法使用
            print("✗ 未找到可用的摄像头")
            self.error_occurred.emit("No available camera found")
            return False
            
        except Exception as e:
            print(f"✗ 摄像头初始化失败: {str(e)}")
            self.error_occurred.emit(f"Camera initialization failed: {str(e)}")
            return False

    def init_visual_recognition(self):
        """初始化视觉识别模块"""
        if not VISUAL_RECOGNITION_AVAILABLE:
            print("⚠️  视觉识别模块不可用，仅启用基础摄像头功能")
            return True
            
        try:
            print("🧠 初始化视觉识别模块...")
            
            if WEIGHT_FILE_PATH:
                # 使用找到的权重文件
                self.visual_interface = VisualRecognitionInterface(
                    model_name="resnet18",
                    weight_path=WEIGHT_FILE_PATH,
                    dataset="mpiigaze"
                )
                print("✓ 使用完整的PyTorch模型")
            else:
                # 使用简化接口
                self.visual_interface = VisualRecognitionInterface(
                    model_name="opencv",
                    weight_path=None,
                    dataset="basic"
                )
                print("✓ 使用OpenCV简化模型")
            
            # 设置回调函数
            if hasattr(self.visual_interface, 'set_distraction_callback'):
                self.visual_interface.set_distraction_callback(self.on_distraction_detected)
            if hasattr(self.visual_interface, 'set_gesture_callback'):
                self.visual_interface.set_gesture_callback(self.on_gesture_detected)
            
            print("✓ 视觉识别系统初始化成功")
            return True
            
        except Exception as e:
            print(f"⚠️  视觉识别初始化失败: {str(e)}")
            self.visual_interface = None
            return True

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
        
        camera_ok = self.init_camera()
        visual_ok = self.init_visual_recognition()
        
        if camera_ok:
            self.running = True
            self.start()
            if WEIGHT_FILE_PATH:
                print("✓ 完整系统启动成功（使用PyTorch模型）")
            else:
                print("✓ 基础系统启动成功（使用OpenCV模型）")
        else:
            print("✗ 系统启动失败：摄像头不可用")
            self.error_occurred.emit("Camera not available")

    def stop_recognition(self):
        """停止识别"""
        print("🛑 停止视觉识别系统...")
        self.running = False

        # 释放摄像头资源
        if self.cap:
            self.cap.release()
            self.cap = None
    
        # 停止线程
        self.quit()
        self.wait(3000)  # 等待最多3秒
        if self.isRunning():
            self.terminate()  # 强制终止
            self.wait(1000)
    
        print("✓ 视觉识别系统已完全停止")

    def run(self):
        """主运行循环"""
        if not self.cap:
            print("✗ 摄像头未初始化")
            return
            
        print("▶️  开始视频处理循环...")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            try:
                # 发送帧数据
                self.frame_ready.emit(frame)
                
                # 处理视觉识别
                if self.visual_interface:
                    result = self.visual_interface.process_frame(frame)
                    self.gaze_data.emit(result)
                else:
                    # 基础模式
                    self.gaze_data.emit({
                        'status': 'camera_only',
                        'message': '纯摄像头模式',
                        'frame_count': self.frame_count
                    })
                
                self.frame_count += 1
                self.msleep(33)  # ~30 FPS
                
            except Exception as e:
                print(f"⚠️  帧处理错误: {e}")
                continue
                
        print("🔚 视频处理循环结束")
        if self.cap:
            self.cap.release()