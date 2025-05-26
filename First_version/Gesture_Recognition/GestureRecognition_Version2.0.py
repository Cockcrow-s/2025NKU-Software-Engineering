# gesture_thread.py
from PyQt5.QtCore import QThread, QCoreApplication
import cv2
import mediapipe as mp
import numpy as np
from collections import deque  # 使用双端队列记录历史位置
import sys


class GestureThread(QThread):
    # 全局记录手腕历史
    HISTORY_MAXLEN = 5
    wrist_history = deque(maxlen=HISTORY_MAXLEN)

    def get_hand_gesture(landmarks):

        # —— 1. 每帧都记录手腕横坐标 —— 
        wrist = landmarks[0]
        GestureThread.wrist_history.append(wrist.x)

        # —— 2. 优先检测大拇指、握拳等静态手势 —— 
        if GestureThread.is_thumb_up(landmarks):
            return "thumb up"
        if GestureThread.is_fist(landmarks):
            return "fist"

        # —— 3. 再看是不是“open palm” —— 
        if GestureThread.is_open_palm(landmarks):
            # 只有当我们已经有足够帧数时，才做“摇手”判断
            if len(GestureThread.wrist_history) >= GestureThread.HISTORY_MAXLEN and GestureThread.is_waving(landmarks):
                return "wave hand"
            else:
                return "palm"

        # —— 4. 其他都算 unknown —— 
        return "unknown"


    def is_waving(landmarks):
        """
        通过横坐标 extremes & 方向变化次数 来判断左右摇手
        """
        xs = list(GestureThread.wrist_history)
        # 真正的掌宽
        palm_width = abs(landmarks[5].x - landmarks[17].x)
        # 只要求移动超过 palm_width * 0.5 即可
        if max(xs) - min(xs) < 0.8 * palm_width:
            return False

        # 检测方向来回变化次数
        changes = 0
        last_dir = 0
        for i in range(1, len(xs)):
            dx = xs[i] - xs[i-1]
            if abs(dx) < 1e-3:
                continue
            dir = 1 if dx > 0 else -1
            if last_dir != 0 and dir != last_dir:
                changes += 1
            last_dir = dir

        # 至少来回一次（changes>=1）
        return changes >= 1

    # 判断是否点赞
    def is_thumb_up(landmarks):
        # 关键点
        thumb_tip = landmarks[4]
        # thumb_base = landmarks[1]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_ip = landmarks[7]
        index_base = landmarks[5]
        middle_base = landmarks[9]
        ring_base = landmarks[13]
        
        # 计算手部基准尺度
        palm_width = abs(landmarks[5].x - landmarks[17].x)
        
        # 条件1：拇指竖直分量
        thumb_vector_y = thumb_ip.y - thumb_tip.y  # 指尖的y更小
        vertical_condition = thumb_vector_y > (palm_width * 0.3)
        
        # 条件2：拇指-食指二端分离
        thumb_index_ip_dist = np.linalg.norm([thumb_tip.x - index_ip.x, thumb_tip.y - index_ip.y])
        separation_condition = thumb_index_ip_dist > (palm_width * 0.5)

        # 条件3：食指尖-其他手指根部距离
        thumb_index_tip_dist = np.linalg.norm([index_tip.x - index_base.x, index_tip.y - index_base.y])
        thumb_index_tip_condition = thumb_index_tip_dist < (palm_width * 1.0)
        index_middle_tip_dist = np.linalg.norm([index_tip.x - middle_base.x, index_tip.y - middle_base.y])
        index_middle_condition = index_middle_tip_dist < (palm_width * 1.0)
        index_dist = np.linalg.norm([ 2*(index_tip.x) - (index_base.x + middle_base.x), 2*(index_tip.y) - (index_base.y + middle_base.y) ])
        index_condition = index_dist < (palm_width * 1.5)
        
        # # 条件3：拇指朝前（可选）
        # z_condition = thumb_tip.z < thumb_ip.z - 0.05
        
        return vertical_condition and separation_condition and (thumb_index_tip_condition or index_middle_condition or index_condition) # and z_condition

    # OK我感觉这里写的很OK了
    def is_fist(landmarks, threshold_scale=0.7):
        # 关键点定义
        fingertips = [4, 8, 12, 16, 20]  # 指尖
        palm_indices = [0, 1, 5, 9, 13, 17]  # 掌心参考点
        
        # 计算掌心（2D）
        palm_points = np.array([(landmarks[i].x, landmarks[i].y) for i in palm_indices])
        palm_center = np.mean(palm_points, axis=0)
        
        # 动态阈值（基于手掌宽度）
        palm_width = abs(landmarks[1].x - landmarks[17].x)
        dynamic_threshold = palm_width * threshold_scale
        
        # 检查所有指尖
        for tip in fingertips:
            tip_pos = np.array([landmarks[tip].x, landmarks[tip].y])
            dist = np.linalg.norm(tip_pos - palm_center)
            if tip == 4:  # 拇指
                if dist > 2*dynamic_threshold:
                    return False
            elif dist > dynamic_threshold:
                return False
        return True

    # 判断是否成手掌
    def is_open_palm(landmarks):
        fingertips = [4, 8, 12, 16, 20]
        palm_center = np.mean([(landmarks[i].x, landmarks[i].y) for i in [0, 5, 9, 13, 17]], axis=0)

        # 动态阈值（基于手掌宽度）
        palm_width = abs(landmarks[1].x - landmarks[17].x)
        dynamic_threshold = palm_width * 1.0

        # 所有指尖远离掌心
        return all(
            np.linalg.norm([landmarks[i].x - palm_center[0], landmarks[i].y - palm_center[1]]) > dynamic_threshold
            for i in fingertips
        )
    
    # 直接调用这个函数，得到手势的识别后的string类型变量
    def get_hand_gesture_by_vedio(cap):

        # 初始化 MediaPipe 手部检测
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
        mp_drawing = mp.solutions.drawing_utils

        max_plam_count = 10  # 检测到手掌超过五次且不摇手则判断为手掌
        temp_palm_count = 0

        while cap.isOpened():
            current_gesture = "unknown"
            ret, frame = cap.read()
            if not ret:
                continue

            # 转换为 RGB 并检测手势
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 绘制关键点
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    # 获取手势类型
                    current_gesture = GestureThread.get_hand_gesture(
                        [lm for lm in hand_landmarks.landmark]
                    )
                    # 语音提示
                    # if current_gesture != "unknown":
                    #     last_gesture = current_gesture
            
            # 这里需要把摇手和其他情况区分开；其他情况直接结束；检测到手掌时判断是否继续摇手
            if current_gesture != "unknown" and current_gesture != "palm":
                break
            elif current_gesture == "palm":
                temp_palm_count += 1
                if temp_palm_count > max_plam_count:
                    break
            else:
                continue

        return current_gesture

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        # 直接调用这个函数，得到手势的识别后的string类型变量
        gesture = GestureThread.get_hand_gesture_by_vedio(cap)
        print(gesture)
        cap.release()
        cv2.destroyAllWindows()        
        pass

def main():
    # 必须创建 QApplication/QCoreApplication 实例
    app = QCoreApplication(sys.argv)
    
    # 创建并启动线程
    thread = GestureThread()
    thread.finished.connect(app.quit)  # 线程结束时退出程序
    thread.start()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()