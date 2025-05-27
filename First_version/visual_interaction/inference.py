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
import mediapipe as mp

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze

import uniface

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')


class HeadGestureRecognizer:
    def __init__(self, frame_queue=None):
        # 初始化MediaPipe人脸网格和参数
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
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

    def get_head_pose(self, landmarks, image_shape):
        """
        根据人脸关键点估算头部姿态（pitch/yaw/roll）
        :param landmarks: MediaPipe人脸关键点
        :param image_shape: 图像尺寸
        :return: pitch, yaw, roll
        """
        image_h, image_w = image_shape[:2]
        # 选取6个人脸关键点
        indices = [1, 33, 263, 61, 291, 199]
        pts = np.array([(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in indices], dtype=np.float32)
        # 3D模型点
        model_points = np.array([
            [0.0, 0.0, 0.0],
            [-30.0, -30.0, -30.0],
            [30.0, -30.0, -30.0],
            [-30.0, 30.0, -30.0],
            [30.0, 30.0, -30.0],
            [0.0, 50.0, -5.0],
        ])
        # 相机内参
        focal_length = image_w
        center = (image_w / 2, image_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        # PnP解算头部姿态
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return None, None, None
        rmat, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        pitch = np.arctan2(-rmat[2, 0], sy)
        yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
        roll = np.arctan2(rmat[2, 1], rmat[2, 2])
        return pitch, yaw, roll

    def detect_gesture(self, pitch, yaw, roll):
        """
        检测点头/摇头动作：
        - 点头：roll上下一个来回且幅度都超过阈值
        - 摇头：pitch左右一个来回且幅度都超过阈值
        :return: 'nod'/'shake'/None
        """
        gesture = None
        # 点头检测
        if self.last_roll is not None:
            delta_roll = roll - self.last_roll
            # roll方向变化状态机
            if not hasattr(self, 'roll_state'):
                self.roll_state = 0  # 0:初始, 1:向上, 2:向下
            if self.roll_state == 0 and delta_roll > self.nod_threshold:
                self.roll_state = 1
                self.roll_peak = roll
            elif self.roll_state == 1 and delta_roll < -self.nod_threshold:
                if abs(self.roll_peak - roll) > self.nod_threshold * 2:
                    gesture = 'nod'
                self.roll_state = 0
            elif self.roll_state == 0 and delta_roll < -self.nod_threshold:
                self.roll_state = 2
                self.roll_peak = roll
            elif self.roll_state == 2 and delta_roll > self.nod_threshold:
                if abs(self.roll_peak - roll) > self.nod_threshold * 2:
                    gesture = 'nod'
                self.roll_state = 0
        self.last_roll = roll
        # 摇头检测
        if self.last_pitch is not None:
            delta_pitch = pitch - self.last_pitch
            if not hasattr(self, 'pitch_state'):
                self.pitch_state = 0  # 0:初始, 1:向上, 2:向下
            if self.pitch_state == 0 and delta_pitch > self.shake_threshold:
                self.pitch_state = 1
                self.pitch_peak = pitch
            elif self.pitch_state == 1 and delta_pitch < -self.shake_threshold:
                if abs(self.pitch_peak - pitch) > self.shake_threshold * 2:
                    gesture = 'shake'
                self.pitch_state = 0
            elif self.pitch_state == 0 and delta_pitch < -self.shake_threshold:
                self.pitch_state = 2
                self.pitch_peak = pitch
            elif self.pitch_state == 2 and delta_pitch > self.shake_threshold:
                if abs(self.pitch_peak - pitch) > self.shake_threshold * 2:
                    gesture = 'shake'
                self.pitch_state = 0
        self.last_pitch = pitch
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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            gesture = None
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                pitch, yaw, roll = self.get_head_pose(landmarks, frame.shape)
                if pitch is not None and yaw is not None:
                    gesture = self.detect_gesture(pitch, yaw, roll)
                    # 显示姿态角度
                    cv2.putText(frame, f"Pitch: {pitch:.2f} Yaw: {yaw:.2f} Roll: {roll:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    # 可视化点头/摇头
                    if gesture == 'nod':
                        cv2.putText(frame, "Nod (Confirm)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    elif gesture == 'shake':
                        cv2.putText(frame, "Shake (Reject)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.imshow('Head Gesture Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


def run_head_gesture(frame_queue, result_queue):
    """
    子线程：从帧队列读取图像，检测头部姿态并返回结果
    :param frame_queue: 输入帧队列
    :param result_queue: 输出结果队列
    """
    recognizer = HeadGestureRecognizer(frame_queue=frame_queue)
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = recognizer.face_mesh.process(rgb)
        gesture = None
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            pitch, yaw, roll = recognizer.get_head_pose(landmarks, frame.shape)
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
    2. 进入主循环，逐帧检测人脸、估计视线、校准安全夹角、分心检测
    3. 实时绘制人脸框、视线向量、安全圈、分心警告和yaw波形图
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

    # 校准状态
    calibration_mode = True
    calibration_angles = []
    safe_angle = None
    calibration_text = "Calibration: Please look forward and press SPACE to finish"
    calibration_progress = 0
    calibration_max = 50
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
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # 计算视线与正前方的3D夹角theta
                theta = np.arccos(np.cos(pitch_predicted) * np.cos(yaw_predicted))
                # 实时显示theta与正前方夹角
                cv2.putText(frame, f"Theta deviation: {theta:+.2f} rad", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                # 记录theta角度到波形历史
                angle_history.append(theta)
                # 绘制安全视线圈（随当前视线向量长度和安全角度动态变化）
                if not calibration_mode and safe_angle is not None:
                    # 以人脸中心为圆心，gaze_len为当前视线向量长度
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2
                    gaze_len = int(0.8 * (y_max - y_min))  # 视线向量长度
                    # 安全圈半径 = gaze_len * sin(safe_angle)
                    safe_radius = int(gaze_len * np.sin(safe_angle))
                    if safe_radius > 5:
                        cv2.circle(frame, (cx, cy), safe_radius, (0, 255, 255), 2)  # 黄色安全圈
                    # 画安全夹角扇形（弧线）
                    start_angle = int(-np.degrees(safe_angle))
                    end_angle = int(np.degrees(safe_angle))
                    cv2.ellipse(frame, (cx, cy), (gaze_len, gaze_len), 0, start_angle, end_angle, (0, 255, 255), 2)

                # 校准逻辑
                if calibration_mode:
                    calibration_angles.append((pitch_predicted, yaw_predicted))  # 收集校准阶段的yaw/pitch
                    calibration_progress += 1
                    # 绘制校准提示文本和进度条
                    cv2.putText(frame, calibration_text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    bar_x, bar_y, bar_w, bar_h = 40, 60, 400, 20
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (255,255,255), 2)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+int(bar_w*min(1,calibration_progress/calibration_max)), bar_y+bar_h), (0,255,0), -1)
                    # 检查是否按下空格键，完成校准
                    key = cv2.waitKey(1)
                    if key == 32:  # 空格键
                        if len(calibration_angles) > 10:
                            # 以3D夹角theta为主，夹角绝对值最大值为安全范围
                            thetas = [np.arccos(np.cos(p)*np.cos(y)) for p, y in calibration_angles]
                            safe_angle = max(abs(np.min(thetas)), abs(np.max(thetas))) 
                            if safe_angle > 0.1:
                                safe_angle = 0.1
                        else:
                            safe_angle = 0.1  # 兜底安全角度
                        calibration_mode = False
                        print(f"Calibration finished. Safe theta angle: {safe_angle:.2f} rad")
                        continue
                    else:
                        continue  # 只做校准，不进入分心检测
                # 分心检测逻辑
                if safe_angle is not None:
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
