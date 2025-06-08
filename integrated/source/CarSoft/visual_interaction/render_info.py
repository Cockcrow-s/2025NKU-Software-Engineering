"""
修复后的render_info_on_frame方法，支持手势识别显示
"""

import cv2
import numpy as np
import time

def render_info_on_frame(self, frame, result):
    """
    在帧上渲染分析信息
    :param frame: 输入帧
    :param result: process_frame返回的结果
    :return: 渲染后的帧
    """
    # 如果有手势识别器，先绘制手部标记
    if self.hand_gesture_recognizer:
        frame = self.hand_gesture_recognizer.draw_landmarks(frame, result.get("gesture"))
    
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
    
    # 显示检测到的手势（如果有）
    if result.get("gesture"):
        gesture_map = {
            'nod': '点头确认',
            'shake': '摇头拒绝',
            'thumbs_up': '大拇指确认安全',
            'wave': '摇手拒绝警告',
            'ok': 'OK手势确认',
            'stop': '停止手势'
        }
        gesture_text = gesture_map.get(result["gesture"], result["gesture"])
        cv2.putText(frame, f"手势: {gesture_text}", (30, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    # 分心警告显示 - 增强的红色边框警告
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
    
    return frame
