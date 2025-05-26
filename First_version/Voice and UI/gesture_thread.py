# gesture_thread.py
from PyQt5.QtCore import QThread

class GestureThread(QThread):
    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        # 预留手势识别线程结构，暂不实现具体功能
        pass
