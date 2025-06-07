import sys
import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QScrollArea, QToolButton, QFileDialog, QMessageBox, QPushButton,
    QProgressBar, QFrame, QGraphicsDropShadowEffect, QGroupBox,
    QGridLayout, QSizePolicy, QTextEdit
)
from PyQt5.QtGui import QIcon, QFont, QPixmap, QColor, QPalette, QLinearGradient, QImage
from PyQt5.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve, pyqtSignal

from music_player import MusicPlayer
from voice_thread import VoiceThread
from gesture_thread import GestureThread
from face_thread import FaceThread
from music_window import MusicWindow
import state
import cv2
import numpy as np


class ModernCard(QFrame):
    """现代化卡片组件"""
    def __init__(self, title="", content="", icon_path=""):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(45, 45, 45, 0.9);
                border: 2px solid rgba(70, 130, 180, 0.3);
                border-radius: 15px;
                padding: 10px;
            }
            QFrame:hover {
                border: 2px solid rgba(70, 130, 180, 0.8);
                background-color: rgba(55, 55, 55, 0.9);
            }
        """)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 5)
        self.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout()
        
        # 标题
        if title:
            title_label = QLabel(title)
            title_label.setFont(QFont("Arial", 14, QFont.Bold))
            title_label.setStyleSheet("color: #4682B4; margin-bottom: 5px;")
            layout.addWidget(title_label)
        
        # 内容
        if content:
            content_label = QLabel(content)
            content_label.setFont(QFont("Arial", 12))
            content_label.setStyleSheet("color: white;")
            content_label.setWordWrap(True)
            layout.addWidget(content_label)
        
        self.setLayout(layout)


class StatusIndicator(QLabel):
    """状态指示器"""
    def __init__(self, text="", color="green"):
        super().__init__(text)
        self.setFont(QFont("Arial", 10, QFont.Bold))
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(100, 30)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: white;
                border-radius: 15px;
                padding: 5px;
            }}
        """)

from system_ui import SystemManagementWindow  # 引入系统管理界面
from settings_ui import SettingsWindow  # 导入设置界面

# 添加父目录到Python路径，以便导入其他模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from System_management.user_info import User,initialize_user_database

class MainWindow(QMainWindow):
    def __init__(self,user):
        super().__init__()
        self.user = user
        self.setWindowTitle("车载多模态智能交互系统 - SmartDrive AI")
        self.setFixedSize(1200, 800)
          # 设置现代化样式
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #0D1B2A, stop: 0.5 #1B263B, stop: 1 #415A77);
            }
        """)
        
        # 可选：设置背景图片（如果存在的话）
        background_path = "resources/background.jpg"
        if os.path.exists(background_path):
            self.set_background(background_path)
        
        # 初始化组件
        self.music_player = MusicPlayer()
        self.init_ui()
        self.init_threads()
        self.setup_timers()
        
    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 顶部状态栏
        self.create_status_bar(main_layout)
        
        # 中间内容区域
        content_layout = QHBoxLayout()
        
        # 左侧面板
        self.create_left_panel(content_layout)
        
        # 中央视频区域
        self.create_video_area(content_layout)
        
        # 右侧控制面板
        self.create_right_panel(content_layout)
        
        main_layout.addLayout(content_layout)
        
        # 底部功能按钮
        self.create_bottom_controls(main_layout)
    
    def create_status_bar(self, parent_layout):
        """创建状态栏"""
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.StyledPanel)
        status_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(13, 27, 42, 0.9);
                border: 2px solid rgba(70, 130, 180, 0.5);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        
        status_layout = QHBoxLayout(status_frame)
        
        # 系统状态
        self.system_status = StatusIndicator("系统正常", "#28A745")
        status_layout.addWidget(self.system_status)
        
        # 视觉监控状态
        self.visual_status = StatusIndicator("视觉监控", "#FFC107")
        status_layout.addWidget(self.visual_status)
        
        # 分心警告
        self.warning_status = StatusIndicator("正常驾驶", "#28A745")
        status_layout.addWidget(self.warning_status)
        
        status_layout.addStretch()
        
        # 时间显示
        self.time_label = QLabel()
        self.time_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.time_label.setStyleSheet("color: #4682B4;")
        status_layout.addWidget(self.time_label)
        
        parent_layout.addWidget(status_frame)
    
    def create_left_panel(self, parent_layout):
        """创建左侧面板"""
        left_panel = QFrame()
        left_panel.setFixedWidth(300)
        left_panel.setStyleSheet("""
            QFrame {
                background-color: rgba(27, 38, 59, 0.8);
                border-radius: 15px;
                padding: 15px;
            }
        """)
        
        left_layout = QVBoxLayout(left_panel)        # 驾驶员信息卡片 - 显示用户信息
        user_info = f"用户: {self.user.username}\n角色: {self.user.role}\n状态: 已登录"
        driver_card = ModernCard("驾驶员信息", user_info)
        left_layout.addWidget(driver_card)
        
        # 系统监控卡片
        self.monitor_card = ModernCard("系统监控", "视线追踪: 待启动\n手势识别: 待启动\n语音识别: 已启动")
        left_layout.addWidget(self.monitor_card)
        
        left_layout.addStretch()
        parent_layout.addWidget(left_panel)
    
    def create_video_area(self, parent_layout):
        """创建视频显示区域"""
        video_frame = QFrame()
        video_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 0.8);
                border: 2px solid rgba(70, 130, 180, 0.5);
                border-radius: 15px;
            }
        """)
        
        video_layout = QVBoxLayout(video_frame)
        
        # 视频显示标签
        self.video_label = QLabel("摄像头视频流")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setStyleSheet("""
            QLabel {
                color: #4682B4;
                font-size: 18px;
                font-weight: bold;
                border: 2px dashed rgba(70, 130, 180, 0.3);
                border-radius: 10px;
            }
        """)
        video_layout.addWidget(self.video_label)
        
        # 视频控制按钮
        video_controls = QHBoxLayout()
        
        self.start_camera_btn = QPushButton("启动摄像头")
        self.start_camera_btn.setStyleSheet("""
            QPushButton {
                background-color: #28A745;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #34CE57;
            }
        """)
        self.start_camera_btn.clicked.connect(self.start_camera)
        video_controls.addWidget(self.start_camera_btn)
        
        self.stop_camera_btn = QPushButton("停止摄像头")
        self.stop_camera_btn.setStyleSheet("""
            QPushButton {
                background-color: #DC3545;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E55A6A;
            }
        """)
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.stop_camera_btn.setEnabled(False)
        video_controls.addWidget(self.stop_camera_btn)
        
        video_controls.addStretch()
        video_layout.addLayout(video_controls)
        
        parent_layout.addWidget(video_frame)
    
    def create_right_panel(self, parent_layout):
        """创建右侧控制面板"""
        right_panel = QFrame()
        right_panel.setFixedWidth(300)
        right_panel.setStyleSheet("""
            QFrame {
                background-color: rgba(27, 38, 59, 0.8);
                border-radius: 15px;
                padding: 15px;
            }
        """)
        
        right_layout = QVBoxLayout(right_panel)
        
        # 警告信息显示
        warning_group = QGroupBox("警告信息")
        warning_group.setStyleSheet("""
            QGroupBox {
                color: #DC3545;
                font-weight: bold;
                border: 2px solid rgba(220, 53, 69, 0.3);
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        warning_layout = QVBoxLayout()
        
        self.warning_text = QTextEdit()
        self.warning_text.setMaximumHeight(100)
        self.warning_text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(45, 45, 45, 0.8);
                color: #DC3545;
                border: 1px solid rgba(220, 53, 69, 0.3);
                border-radius: 5px;
                padding: 5px;
                font-family: 'Consolas';
            }
        """)
        self.warning_text.setPlainText("系统正常运行，无警告信息")
        warning_layout.addWidget(self.warning_text)
        
        warning_group.setLayout(warning_layout)
        right_layout.addWidget(warning_group)
        
        # 手势识别信息
        gesture_card = ModernCard("手势识别", "当前手势: 无\n上次识别: --")
        self.gesture_info = gesture_card
        right_layout.addWidget(gesture_card)
        
        # 语音交互信息
        voice_card = ModernCard("语音交互", "状态: 监听中\n最后命令: --")
        self.voice_info = voice_card
        right_layout.addWidget(voice_card)
        
        right_layout.addStretch()
        parent_layout.addWidget(right_panel)
    
    def create_bottom_controls(self, parent_layout):
        """创建底部控制按钮"""
        controls_frame = QFrame()
        controls_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(13, 27, 42, 0.9);
                border: 2px solid rgba(70, 130, 180, 0.5);
                border-radius: 10px;
                padding: 15px;
            }
        """)
        
        controls_layout = QHBoxLayout(controls_frame)
          # 功能按钮 - 根据用户角色显示不同按钮
        buttons = [
            ("音乐播放", "🎵", self.open_music),
            ("天气信息", "🌤️", self.open_weather),
            ("系统设置", "⚙️", self.open_settings)
        ]
        
        # 只有管理员才能看到系统管理按钮
        if hasattr(self.user, 'role') and self.user.role == "admin":
            buttons.append(("系统管理", "🔧", self.open_system))
        
        for text, icon, callback in buttons:
            btn = QPushButton(f"{icon} {text}")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(70, 130, 180, 0.8);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    padding: 15px 25px;
                    font-size: 14px;
                    font-weight: bold;
                    min-width: 120px;                }
                QPushButton:hover {
                    background-color: rgba(70, 130, 180, 1.0);
                }
                QPushButton:pressed {
                    background-color: rgba(47, 95, 143, 1.0);
                }
            """)
            btn.clicked.connect(callback)
            controls_layout.addWidget(btn)
        
        parent_layout.addWidget(controls_frame)
    def init_threads(self):
        """初始化线程"""
        print("\n🔧 初始化系统线程...")
          # 语音线程
        print("🎤 启动语音识别线程...")
        self.voice_thread = VoiceThread(self.music_player)
        self.voice_thread.safety_confirmed.connect(self.on_safety_confirmed_by_voice)
        self.voice_thread.start()
        print("✓ 语音识别系统已启动")
        
        # 手势线程
        print("👋 启动手势识别线程...")
        self.gesture_thread = GestureThread()
        self.gesture_thread.gesture_recognized.connect(self.on_gesture_recognized)
        self.gesture_thread.gesture_action.connect(self.on_gesture_action)
        self.gesture_thread.start_gesture_recognition()
        print("✓ 手势识别系统已启动")
        
        # 面部识别线程
        print("👁️  启动面部识别线程...")
        self.face_thread = FaceThread()
        self.face_thread.frame_ready.connect(self.update_video_display)
        self.face_thread.distraction_detected.connect(self.on_distraction_detected)
        self.face_thread.gesture_detected.connect(self.gesture_thread.process_gesture)
        self.face_thread.gaze_data.connect(self.on_gaze_data_received)
        self.face_thread.error_occurred.connect(self.on_error_occurred)
        print("✓ 面部识别系统已启动")
        
        print("🎯 三模态交互系统初始化完成！")
    
    def setup_timers(self):
        """设置定时器"""
        # 时间更新定时器
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)
        
        # 状态更新定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(500)
        
        # 警告闪烁定时器
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.blink_warning)
        self.blink_state = True
    
    def update_time(self):
        """更新时间显示"""
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(current_time)
    
    def update_status(self):
        """更新状态显示"""
        status = state.get_system_status()        # 更新视觉状态
        if status['visual_state'] == 'monitoring':
            self.visual_status.setText("监控中")
            self.visual_status.setStyleSheet("""
                QLabel {
                    background-color: #28A745;
                    color: white;
                    border-radius: 15px;
                    padding: 5px;
                }
            """)
        
        # 更新警告状态
        if status['is_warning']:
            self.warning_status.setText("分心警告")
            self.warning_status.setStyleSheet("""
                QLabel {
                    background-color: #DC3545;
                    color: white;
                    border-radius: 15px;
                    padding: 5px;
                }
            """)
            if not self.blink_timer.isActive():
                self.blink_timer.start(500)
        else:
            self.warning_status.setText("正常驾驶")
            self.warning_status.setStyleSheet("""
                QLabel {
                    background-color: #28A745;
                    color: white;
                    border-radius: 15px;
                    padding: 5px;
                }
            """)
            self.blink_timer.stop()
    
    def blink_warning(self):
        """警告闪烁效果"""
        if state.is_warning:
            if self.blink_state:                self.warning_status.setStyleSheet("""
                    QLabel {
                        background-color: #FF6B6B;
                        color: white;
                        border-radius: 15px;
                        padding: 5px;
                    }
                """)
            else:
                self.warning_status.setStyleSheet("""
                    QLabel {
                        background-color: #DC3545;
                        color: white;
                        border-radius: 15px;
                        padding: 5px;
                    }                """)
            self.blink_state = not self.blink_state
    
    def start_camera(self):
        """启动摄像头"""
        print("📹 启动摄像头和视觉识别系统...")
        self.face_thread.start_recognition()
        self.start_camera_btn.setEnabled(False)
        self.stop_camera_btn.setEnabled(True)
        self.visual_status.setText("运行中")
        print("✓ 摄像头已启动，开始视觉监控")
    
    def stop_camera(self):
        """停止摄像头"""
        print("⏹️  停止摄像头和视觉识别系统...")
        self.face_thread.stop_recognition()
        self.start_camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)
        self.visual_status.setText("已停止")
        self.video_label.setText("摄像头视频流")
        print("✓ 摄像头已停止")
    
    def update_video_display(self, frame):
        """更新视频显示"""
        try:
            # 转换OpenCV图像到Qt格式
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
              # 缩放图像以适应显示区域
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Video display error: {e}")
    
    def on_distraction_detected(self, distracted, warning_level):
        """处理分心检测 - 三模态异常反馈场景"""
        if distracted:
            print(f"⚠️  检测到分心驾驶！警告级别: {warning_level}")
            
            # 1. 文本反馈 - 状态栏显示红色闪烁
            if warning_level <= 2:
                warning_msg = "警告！请目视前方"
                self.warning_text.setPlainText(f"⚠️  {warning_msg}\n警告级别: {warning_level}\n请立即注意道路安全！")
            else:
                warning_msg = "危险！立即目视前方！"
                self.warning_text.setPlainText(f"🚨 {warning_msg}\n警告级别: {warning_level}\n危险等级升级！")
            
            # 2. 语音反馈 - 根据警告级别播报不同内容
            if warning_level <= 2:
                print("🔊 语音播报：请注意行车安全")
                # 这里可以添加TTS语音播报功能
            else:
                print("🔊 语音播报：请立即目视前方！")
                # 升级语音警告
            
            # 3. 视觉提示 - 启动状态栏红色闪烁
            self.start_warning_blink()
            
            # 等待驾驶员确认 - 可通过语音说"已注意道路"或手势确认            print("👋 等待驾驶员确认：说'已注意道路'或竖起大拇指确认")
            
        else:
            print("✓ 驾驶状态正常")
            self.warning_text.setPlainText("✓ 系统正常运行，驾驶状态良好")
            self.stop_warning_blink()
    
    def on_gesture_recognized(self, gesture):
        """处理手势识别 - 支持安全确认手势"""
        gesture_map = {
            'nod': '点头确认',
            'shake': '摇头拒绝',
            'thumbs_up': '大拇指确认安全',
            'wave': '摇手拒绝警告',
            'ok': '手势确认'
        }
        
        gesture_text = gesture_map.get(gesture, gesture)
        print(f"👋 检测到手势: {gesture_text}")
        
        # 处理安全确认手势
        if gesture in ['thumbs_up', 'ok', 'nod']:
            print("✓ 驾驶员通过手势确认安全状态")
            self.on_safety_confirmed_by_gesture()
        elif gesture in ['wave', 'shake']:
            print("❌ 驾驶员拒绝警告确认")
            self.warning_text.setPlainText("⚠️  驾驶员拒绝确认，请继续保持警惕！")
          # 更新手势信息显示
        new_content = f"当前手势: {gesture_text}\n上次识别: 刚刚\n状态: 已处理"
    
    def on_gesture_action(self, action):
        """处理手势动作 - 三模态交互场景"""
        print(f"🎯 处理手势动作: {action}")
        
        if action == "attention_confirmed":
            self.on_safety_confirmed_by_gesture()
        elif action == "safety_confirmed":
            self.on_safety_confirmed_by_gesture()
        elif action == "warning_rejected":
            print("❌ 驾驶员拒绝警告确认")
            self.warning_text.setPlainText("⚠️  驾驶员拒绝确认警告，请继续保持高度警惕！")
        else:
            print(f"📝 其他手势动作: {action}")
            self.warning_text.setPlainText(f"检测到手势动作: {action}")
    
    def on_gaze_data_received(self, data):
        """处理视线数据"""
        # 更新监控卡片信息
        if data.get('theta') is not None:
            monitor_text = f"视线追踪: 运行中\n视线偏移: {data['theta']:.2f}rad\n手势识别: 运行中\n语音识别: 已启动"
            # 更新monitor_card内容
    
    def on_error_occurred(self, error_msg):
        """处理错误"""
        self.warning_text.setPlainText(f"系统错误: {error_msg}")
    
    # 功能按钮回调
    def open_music(self):
        """打开音乐窗口"""
        self.music_window = MusicWindow(self.music_player)
        self.music_window.show()
    
    def open_weather(self):
        """打开天气窗口"""
        QMessageBox.information(self, "天气信息", "天气功能即将上线，敬请期待！")
    
    def open_settings(self):
        """打开设置窗口"""
        try:
            self.settings_window = SettingsWindow(self.user, self)
            self.settings_window.show()
        except Exception as e:
            QMessageBox.information(self, "系统设置", f"设置功能加载中: {str(e)}")
    
    def open_system(self):
        """打开系统管理窗口"""
        try:
            if hasattr(self.user, 'role') and self.user.role == "admin":
                self.system_management_window = SystemManagementWindow(self.user)
                self.system_management_window.show()
            else:
                QMessageBox.warning(self, "权限不足", "只有管理员才能访问系统管理功能！")
        except Exception as e:
            QMessageBox.information(self, "系统管理", f"系统管理功能加载中: {str(e)}")
    
    def set_background(self, image_path):
        """设置背景图片"""
        try:
            normalized_path = os.path.normpath(image_path).replace("\\", "/")
            self.setStyleSheet(f"""
                QMainWindow {{
                    border-image: url("{normalized_path}") 0 0 0 0 stretch stretch;
                }}            """)
        except Exception as e:
            print(f"设置背景图片失败: {e}")
    
    def closeEvent(self, event):
        """关闭事件处理"""
        # 停止所有线程
        if hasattr(self, 'face_thread'):
            self.face_thread.stop_recognition()
        if hasattr(self, 'gesture_thread'):
            self.gesture_thread.stop_gesture_recognition()
        if hasattr(self, 'voice_thread'):
            self.voice_thread.quit()
            self.voice_thread.wait()
        
        event.accept()
    
    def on_safety_confirmed_by_gesture(self):
        """通过手势确认安全状态"""
        print("✅ 安全状态已通过手势确认")
        self.warning_text.setPlainText("✅ 驾驶员已通过手势确认安全状态，警告解除")
        self.stop_warning_blink()
    
    def on_safety_confirmed_by_voice(self):
        """通过语音确认安全状态"""
        print("✅ 安全状态已通过语音确认")
        self.warning_text.setPlainText("✅ 驾驶员已说'已注意道路'，警告解除")
        self.stop_warning_blink()
    
    def start_warning_blink(self):
        """启动警告闪烁"""
        if not hasattr(self, 'warning_blink_timer'):
            self.warning_blink_timer = QTimer()
            self.warning_blink_timer.timeout.connect(self.toggle_warning_blink)
            self.blink_state = False
        
        # 根据危险等级调整闪烁频率
        self.warning_blink_timer.start(500)  # 每500ms闪烁一次
        print("🚨 启动仪表盘警告灯闪烁")
    
    def stop_warning_blink(self):
        """停止警告闪烁"""
        if hasattr(self, 'warning_blink_timer'):
            self.warning_blink_timer.stop()
        
        # 恢复正常状态颜色
        self.warning_status.setStyleSheet("""
            QLabel {
                background-color: #28A745;
                color: white;
                border-radius: 15px;
                padding: 5px;
            }
        """)
        self.warning_status.setText("正常驾驶")
        print("✓ 警告灯闪烁已停止")
    
    def toggle_warning_blink(self):
        """切换警告闪烁状态"""
        if self.blink_state:
            # 红色高亮状态
            self.warning_status.setStyleSheet("""
                QLabel {
                    background-color: #FF0000;
                    color: white;
                    border-radius: 15px;
                    padding: 5px;
                    border: 2px solid #FFFFFF;
                }
            """)
            self.warning_status.setText("⚠️ 分心警告")
        else:
            # 深红色状态
            self.warning_status.setStyleSheet("""
                QLabel {
                    background-color: #800000;
                    color: white;
                    border-radius: 15px;
                    padding: 5px;
                }
            """)
            self.warning_status.setText("🚨 注意前方")
        
        self.blink_state = not self.blink_state
