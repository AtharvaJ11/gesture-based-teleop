#!/usr/bin/env python3
"""
ROS2 Hand Tracking Node with Normalized Coordinates
"""
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
import time
import subprocess
import sys


class HandTrackingNode(Node):
    def __init__(self):
        super().__init__('hand_tracking_node')
        
        # ROS2 Configuration
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.bridge = CvBridge()
        
        # Parameters
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('grid_size', 8)

        # Publishers
        self.position_pub = self.create_publisher(Point, 'hand_tracking/position', qos_profile)
        self.debug_pub = self.create_publisher(Image, 'hand_tracking/debug_video', qos_profile)

        # Tracking initialization
        self.cap = cv2.VideoCapture(self.get_parameter('camera_index').value)
        self.trk = HandTracker()
        self.kalman = KalmanTracker()
        
        # Thread management
        self.running = True
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.start()
        
        self.cnt = 0.0

    def process_frames(self):
        rate = self.get_parameter('publish_rate').value
        grid_size = self.get_parameter('grid_size').value
        frame_delay = 1.0 / rate if rate > 0 else 0.0
        quit_hover_start = None
        
        while self.running and self.cap.isOpened():
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("Camera frame read failed")
                continue

            frame = cv2.flip(frame, 1)
            self.trk.detect(frame, self.trk.roi(frame.shape) or None)
            tip = self.trk.fingertip(frame.shape)

            fingertip_x, fingertip_y = None, None

            if tip:
                fingertip_x, fingertip_y = tip
                self.kalman.update(*tip)
                x_kalman, y_kalman = self.kalman.predict()
                norm_x, norm_y = self.publish_position(x_kalman, y_kalman, frame.shape)
                cv2.circle(frame, tip, 8, (0, 255, 0), -1)
                cv2.circle(frame, (int(x_kalman), int(y_kalman)), 8, (0, 0, 255), -1)
                cv2.putText(frame, f"X: {norm_y:.2f} Y: {norm_x:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            draw_grid(frame, grid_size)

            quit_box = (20, 20, 150, 70)
            draw_button(frame, "Quit", quit_box, state="idle")

            if fingertip_x is not None and fingertip_y is not None:
                if is_point_in_box(fingertip_x, fingertip_y, quit_box):
                    if quit_hover_start is None:
                        quit_hover_start = time.time()
                    elif time.time() - quit_hover_start >= HOVER_TIME:
                        print("Quit button selected. Returning to robot_menu...")
                        self.running = False
                        
                        break
                else:
                    quit_hover_start = None

            cv2.imshow("Hand Tracking Debug", frame)
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding="bgr8"))

            elapsed = time.time() - start_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

            if cv2.waitKey(1) == 27:
                self.running = False

    def publish_position(self, x_pixel, y_pixel, frame_shape):
        W = frame_shape[1]
        H = frame_shape[0]
        norm_x = (x_pixel / W) * 2 - 1
        norm_y = 1 - (y_pixel / H) * 2

        msg = Point()
        msg.x = float(norm_x)
        msg.y = float(norm_y)
        msg.z = self.cnt
        self.cnt += 1
        self.position_pub.publish(msg)
        self.get_logger().info(f"Published: X={msg.x:.3f}, Y={msg.y:.3f}, Z={msg.z:.3f}")
        return norm_x, norm_y

    def destroy_node(self):
        self.running = False
        self.processing_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def draw_grid(frame, grid_size):
    H, W = frame.shape[:2]
    stepX, stepY = W // grid_size, H // grid_size

    for x in range(stepX, W, stepX):
        cv2.line(frame, (x, 0), (x, H), (170, 170, 170), 1)
    for y in range(stepY, H, stepY):
        cv2.line(frame, (0, y), (W, y), (170, 170, 170), 1)

    center_x = W // 2
    center_y = H // 2
    cv2.rectangle(frame, (center_x - stepX, center_y - stepY), (center_x + stepX, center_y + stepY), (120, 120, 120), 1)


HOVER_TIME = 3.0
COLOR_IDLE = (200, 200, 200)
COLOR_HOVER = (50, 200, 255)
COLOR_SELECTED = (0, 255, 0)

def draw_button(frame, text, box, state="idle"):
    (x1, y1, x2, y2) = box
    color = COLOR_IDLE if state == "idle" else COLOR_HOVER if state == "hover" else COLOR_SELECTED

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    center_x = x1 + (x2 - x1) // 2
    center_y = y1 + (y2 - y1) // 2

    text_x = center_x - text_w // 2
    text_y = center_y + text_h // 2

    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

def is_point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return (x1 <= px <= x2) and (y1 <= py <= y2)


class KalmanTracker:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.filter = cv2.KalmanFilter(6, 2)
        self.filter.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)
        self.filter.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], np.float32)
        self.filter.processNoiseCov = np.diag([
            1e-4, 1e-4, 1e-2, 1e-2, 5e-1, 5e-1
        ]).astype(np.float32)
        self.filter.measurementNoiseCov = 1e-4 * np.eye(2, dtype=np.float32)
        self.filter.errorCovPost = np.eye(6, dtype=np.float32)

    def update(self, x, y):
        measurement = np.array([[x], [y]], dtype=np.float32)
        self.filter.correct(measurement)

    def predict(self):
        state = self.filter.predict()
        return state[0][0], state[1][0]


class HandTracker:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            model_complexity=1, max_num_hands=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.7)
        self.landmarks = None
        self.last_seen = 0

    def detect(self, frame, roi=None):
        if roi:
            x0, y0, x1, y1 = roi
            crop = frame[y0:y1, x0:x1]
            res = self.hands.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                h, w = frame.shape[:2]
                cw, ch = (x1 - x0), (y1 - y0)
                for p in lm.landmark:
                    p.x = (p.x * cw + x0) / w
                    p.y = (p.y * ch + y0) / h
                self.landmarks = lm
                self.last_seen = time.time()
                return True
        res = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            self.landmarks = res.multi_hand_landmarks[0]
            self.last_seen = time.time()
            return True
        return False

    def fingertip(self, shape):
        if not self.landmarks:
            return None
        h, w = shape[:2]
        tip = self.landmarks.landmark[8]
        return int(tip.x * w), int(tip.y * h)

    def roi(self, shape):
        if not self.landmarks:
            return None
        xs = [p.x for p in self.landmarks.landmark]
        ys = [p.y for p in self.landmarks.landmark]
        x0 = max(min(xs) - ROI_PAD, 0)
        x1 = min(max(xs) + ROI_PAD, 1)
        y0 = max(min(ys) - ROI_PAD, 0)
        y1 = min(max(ys) + ROI_PAD, 1)
        h, w = shape[:2]
        return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


ROI_PAD = 0.25


def main(args=None):
    rclpy.init(args=args)
    node = HandTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        
        rclpy.shutdown()
        


if __name__ == '__main__':
    main()
