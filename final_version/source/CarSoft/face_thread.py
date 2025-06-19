import sys
import os
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..', 'visual_interaction'))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from System_management.interaction_logger import InteractionLogger

try:
    import state
except ImportError:
    print("⚠️  创建临时state模块")
    class TempState:
        is_warning = False
        is_playing = False
    state = TempState()

try:
    from hand_gesture import HandGestureRecognizer
    HAND_GESTURE_AVAILABLE = True
    print("✓ 手势识别模块可用")
except ImportError as e:
    print(f"⚠️  手势识别模块不可用: {e}")
    HAND_GESTURE_AVAILABLE = False

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

VISUAL_RECOGNITION_AVAILABLE = False
VisualRecognitionInterface = None
WEIGHT_FILE_PATH = find_weight_file()

try:
    if WEIGHT_FILE_PATH:
        import torch
        from inference import VisualRecognitionInterface
        VISUAL_RECOGNITION_AVAILABLE = True
    else:
        # 权重文件不存在，使用简化接口
        raise ImportError("Weight file not found")
        
except ImportError as e:
    print(f"⚠️  完整接口不可用: {e}")
    print("使用简化视觉识别接口...")
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
                self.detection_available = True
            except Exception as e:
                self.detection_available = False
            
            # 分心检测相关变量
            self.face_history = []
            self.no_face_count = 0
            self.distraction_count = 0
        
        def set_distraction_callback(self, callback):
            self.distraction_callback = callback
        
        def set_gesture_callback(self, callback):
            self.gesture_callback = callback
        
        def process_frame(self, frame):
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
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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


class FaceThread(QThread):
    # 定义信号
    frame_ready = pyqtSignal(np.ndarray)
    distraction_detected = pyqtSignal(bool, int)
    gesture_detected = pyqtSignal(str)
    gaze_data = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, user=None, parent=None, camera_index=0):
        super().__init__(parent)
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.visual_interface = None
        self.frame_count = 0
        self.camera_available = False
        self.visual_recognition_available = VISUAL_RECOGNITION_AVAILABLE
        
        # 用户信息和日志记录器
        self.current_user = user.username if user else "unknown"
        self.logger = InteractionLogger()
        
        # 分心检测状态追踪
        self.last_warning_level = 0
        self.last_distraction_state = False
        self.detection_session_id = 0
        
        # 初始化手势识别器
        if HAND_GESTURE_AVAILABLE:
            self.hand_gesture_recognizer = HandGestureRecognizer()
            print("✓ 手势识别器初始化成功")
        else:
            self.hand_gesture_recognizer = None
            print("⚠️  手势识别器不可用")
        
        print(f"✓ 面部识别线程初始化完成，用户: {self.current_user}")

    def init_camera(self):
        """初始化摄像头"""
        try:
            print("检测可用摄像头...")
            self.logger.log_interaction(
                self.current_user,
                "系统状态",
                "开始初始化摄像头",
                "尝试检测可用摄像头设备"
            )
            
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
                        
                        # 记录成功初始化日志
                        self.logger.log_interaction(
                            self.current_user,
                            "系统状态",
                            f"摄像头初始化成功 - 索引: {index}, 分辨率: {frame.shape[1]}x{frame.shape[0]}",
                            "摄像头设备可用，视觉监控系统就绪"
                        )
                        
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
            print("未找到可用的摄像头")
            self.logger.log_interaction(
                self.current_user,
                "系统错误",
                "摄像头初始化失败 - 未找到可用设备",
                "视觉监控系统无法启动"
            )
            self.error_occurred.emit("No available camera found")
            return False
            
        except Exception as e:
            print(f"摄像头初始化失败: {str(e)}")
            self.logger.log_interaction(
                self.current_user,
                "系统错误",
                f"摄像头初始化异常: {str(e)}",
                "摄像头设备访问失败"
            )
            self.error_occurred.emit(f"Camera initialization failed: {str(e)}")
            return False

    def init_visual_recognition(self):
        """初始化视觉识别模块"""
        if not VISUAL_RECOGNITION_AVAILABLE:
            self.logger.log_interaction(
                self.current_user,
                "系统状态",
                "视觉识别模块不可用",
                "使用基础摄像头功能"
            )
            return True
            
        try:
            print("初始化视觉识别模块...")
            
            if WEIGHT_FILE_PATH:
                # 使用找到的权重文件
                self.visual_interface = VisualRecognitionInterface(
                    model_name="resnet18",
                    weight_path=WEIGHT_FILE_PATH,
                    dataset="mpiigaze"
                )
                self.logger.log_interaction(
                    self.current_user,
                    "系统状态",
                    f"视觉识别模块初始化 - PyTorch模型, 权重文件: {WEIGHT_FILE_PATH}",
                    "完整AI视觉监控系统启动"
                )
            else:
                self.visual_interface = VisualRecognitionInterface(
                    model_name="opencv",
                    weight_path=None,
                    dataset="basic"
                )
                print("✓ 使用OpenCV简化模型")
                self.logger.log_interaction(
                    self.current_user,
                    "系统状态",
                    "视觉识别模块初始化 - OpenCV简化模型",
                    "基础视觉监控系统启动"
                )
            if hasattr(self.visual_interface, 'set_distraction_callback'):
                self.visual_interface.set_distraction_callback(self.on_distraction_detected)
            if hasattr(self.visual_interface, 'set_gesture_callback'):
                self.visual_interface.set_gesture_callback(self.on_gesture_detected)
            
            print("视觉识别系统初始化成功")
            return True
            
        except Exception as e:
            print(f"视觉识别初始化失败: {str(e)}")
            self.logger.log_interaction(
                self.current_user,
                "系统错误",
                f"视觉识别模块初始化失败: {str(e)}",
                "降级到基础摄像头模式"
            )
            self.visual_interface = None
            return True

    def on_distraction_detected(self, distracted, warning_level):
        # 只有当状态发生变化时才记录日志
        if distracted != self.last_distraction_state or warning_level != self.last_warning_level:
            if distracted:
                warning_text = {
                    1: "轻微分心",
                    2: "中度分心", 
                    3: "严重分心"
                }.get(warning_level, f"未知级别({warning_level})")
                
                self.logger.log_interaction(
                    self.current_user,
                    "安全警告",
                    f"检测到分心驾驶 - {warning_text}",
                    f"触发{warning_level}级安全警告，需要驾驶员确认"
                )
                print(f"分心检测: {warning_text}")
            else:
                self.logger.log_interaction(
                    self.current_user,
                    "安全监控",
                    "驾驶状态恢复正常",
                    "分心状态解除，继续正常监控"
                )
                print("驾驶状态正常")
            
            self.last_distraction_state = distracted
            self.last_warning_level = warning_level
        
        state.is_warning = distracted
        self.distraction_detected.emit(distracted, warning_level)
        
    def on_gesture_detected(self, gesture):
        self.logger.log_interaction(
            self.current_user,
            "视觉识别",
            f"检测到手势: {gesture}",
            "手势识别成功，转发到处理模块"
        )
        self.gesture_detected.emit(gesture)

    def start_recognition(self):
        """开始识别"""
        print("启动车载智能交互系统...")
        self.detection_session_id += 1
        
        # 记录系统启动日志
        self.logger.log_interaction(
            self.current_user,
            "系统状态",
            f"启动视觉识别系统 - 会话ID: {self.detection_session_id}",
            "开始车载智能交互系统"
        )
        
        camera_ok = self.init_camera()
        visual_ok = self.init_visual_recognition()
        
        if camera_ok:
            self.running = True
            self.start()
            if WEIGHT_FILE_PATH:
                self.logger.log_interaction(
                    self.current_user,
                    "系统状态",
                    "完整视觉识别系统启动成功",
                    "AI驱动的安全监控系统已就绪"
                )
            else:
                self.logger.log_interaction(
                    self.current_user,
                    "系统状态",
                    "基础视觉识别系统启动成功",
                    "OpenCV驱动的安全监控系统已就绪"
                )
        else:
            self.logger.log_interaction(
                self.current_user,
                "系统错误",
                "视觉识别系统启动失败",
                "摄像头设备不可用，系统无法启动"
            )
            self.error_occurred.emit("Camera not available")

    def stop_recognition(self):
        """停止识别"""
        print("停止视觉识别系统...")
        
        # 记录系统停止日志
        self.logger.log_interaction(
            self.current_user,
            "系统状态",
            f"停止视觉识别系统 - 会话ID: {self.detection_session_id}",
            f"处理了 {self.frame_count} 帧图像，系统正常关闭"
        )
        
        self.running = False

        # 释放摄像头资源
        if self.cap:
            self.cap.release()
            self.cap = None
    
        # 停止线程
        self.quit()
        self.wait(3000)  
        if self.isRunning():
            self.terminate()  
            self.wait(1000)

    def run(self):
        """主运行循环"""
        if not self.cap:
            self.logger.log_interaction(
                self.current_user,
                "系统错误",
                "视觉识别主循环启动失败",
                "摄像头未正确初始化"
            )
            return
            
        print("开始视频处理循环...")
        
        # 记录处理循环开始
        self.logger.log_interaction(
            self.current_user,
            "系统状态",
            "视频处理循环开始",
            "开始实时视觉监控和分析"
        )
        
        frame_error_count = 0
        max_consecutive_errors = 10
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                frame_error_count += 1
                if frame_error_count >= max_consecutive_errors:
                    self.logger.log_interaction(
                        self.current_user,
                        "系统错误",
                        f"连续 {max_consecutive_errors} 次帧读取失败",
                        "摄像头可能断开连接"
                    )
                    self.error_occurred.emit("Camera disconnected")
                    break
                continue
            else:
                frame_error_count = 0  # 重置错误计数
                
            try:
                # 发送帧数据
                self.frame_ready.emit(frame)
                
                # 处理视觉识别
                if self.visual_interface:
                    result = self.visual_interface.process_frame(frame)
                    self.gaze_data.emit(result)
                    
                    # 每1000帧记录一次处理状态
                    if self.frame_count % 1000 == 0 and self.frame_count > 0:
                        self.logger.log_interaction(
                            self.current_user,
                            "系统监控",
                            f"视觉识别系统运行正常 - 已处理 {self.frame_count} 帧",
                            f"当前状态: {result.get('status', 'unknown')}"
                        )
                else:
                    # 基础模式
                    self.gaze_data.emit({
                        'status': 'camera_only',
                        'message': '纯摄像头模式',
                        'frame_count': self.frame_count
                    })
                
                # 处理手势识别
                if self.hand_gesture_recognizer:
                    try:
                        gesture = self.hand_gesture_recognizer.detect_gesture(frame)
                        if gesture:
                            self.gesture_detected.emit(gesture)
                    except Exception as e:
                        print(f"手势识别错误: {e}")
                        # 每100次手势识别错误才记录一次，避免日志过多
                        if self.frame_count % 100 == 0:
                            self.logger.log_interaction(
                                self.current_user,
                                "系统警告",
                                f"手势识别模块异常: {str(e)}",
                                "手势识别功能可能不稳定"
                            )
                
                self.frame_count += 1
                self.msleep(33)  # ~30 FPS
                
            except Exception as e:
                print(f"帧处理错误: {e}")
                self.logger.log_interaction(
                    self.current_user,
                    "系统错误",
                    f"视频帧处理异常: {str(e)}",
                    "跳过当前帧，继续处理"
                )
                continue
                
        print("视频处理循环结束")
        
        # 记录处理循环结束
        self.logger.log_interaction(
            self.current_user,
            "系统状态",
            f"视频处理循环结束 - 总处理帧数: {self.frame_count}",
            "视觉识别系统正常关闭"
        )
        
        if self.cap:
            self.cap.release()