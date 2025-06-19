import threading
from enum import Enum

# 音乐播放状态
is_playing = False  # 控制音乐播放

# 警告状态
is_warning = False  # 表明是否分心驾驶
warning_level = 0   # 警告级别 0-无警告, 1-轻微, 2-严重

# 视觉识别状态
class VisualState(Enum):
    IDLE = "idle"
    MONITORING = "monitoring"
    WARNING = "warning"

visual_state = VisualState.IDLE

# 手势识别状态
last_gesture = None
gesture_enabled = True

# 语音识别状态
voice_enabled = True
listening = False

# 线程安全锁
state_lock = threading.Lock()

# 用户状态
current_user = None
login_status = False

# 界面状态
ui_theme = "dark"  # 界面主题
show_debug_info = False  # 是否显示调试信息

def update_warning_state(warning, level=0):
    """更新警告状态"""
    global is_warning, warning_level, visual_state
    with state_lock:
        is_warning = warning
        warning_level = level
        if warning:
            visual_state = VisualState.WARNING
        else:
            visual_state = VisualState.MONITORING

def update_visual_state(new_state):
    """更新视觉识别状态"""
    global visual_state
    with state_lock:
        visual_state = new_state

def get_system_status():
    """获取系统状态摘要"""
    with state_lock:
        return {
            'visual_state': visual_state.value,
            'is_warning': is_warning,
            'warning_level': warning_level,
            'is_playing': is_playing,
            'voice_enabled': voice_enabled,
            'gesture_enabled': gesture_enabled,
            'login_status': login_status,
            'current_user': current_user
        }
