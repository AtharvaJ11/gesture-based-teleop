#!/usr/bin/env python3
"""
Authors: Atharva Jamsandekar, Shubhankar Katekari and Nikhil Gutlapalli
Date: 04-23-2025

This program uses ROS2 and Mediapipe to control a robotic arm and gripper
based on human gestures. It processes camera input to classify arm and hand
movements, publishing commands to control joints. The system supports multiple
modes, including base movement, arm positioning, and gripper control.

Additional Feature:
- Detect **two open palms**:
  1) Properly shut down the OpenCV window (and release camera),
  2) THEN publish `True` on /joint_status,
  3) Keep the node alive, waiting for a new /selected_joint to restart camera.

Also:
- For 'arm_up_down' and 'arm_stretch_retract', map the posture into values in [0.0..1.0]
  in increments of 0.1, then publish to the corresponding joint topics.
"""

import cv2
import mediapipe as mp
import math
import time
from collections import deque

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float64, Bool   # Bool for /joint_status
from geometry_msgs.msg import Twist


class ArmFingerNode(Node):
    """
    A robust arm detection approach, plus "gripper" detection and "TWO open palms" to quit camera:
      - 'arm_up_down': publishes a Float64 in [0..1], in steps of 0.1, to /joint_lift
      - 'arm_stretch_retract': publishes a Float64 in [0..1], in steps of 0.1, to
         /joint_arm_l3, /joint_arm_l2, /joint_arm_l1, /joint_arm_l0
      - 'gripper': simple thumb-index distance => open/close
      - 'base': moves /cmd_vel based on fingertip location
      - TWO open palms => camera feed stops, /joint_status => True
    """

    def __init__(self):
        super().__init__('arm_finger_node')

        # Subscriptions
        self.selected_joint_sub = self.create_subscription(
            String,
            '/selected_joint',
            self.joint_callback,
            10
        )
        self.gesture_sub = self.create_subscription(
            String,
            '/gesture_commands',
            self.gesture_callback,
            10
        )

        # Publishers
        self.gesture_pub = self.create_publisher(String, '/gesture_commands', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.joint_lift_pub    = self.create_publisher(Float64, '/joint_lift', 10)
        self.joint_arm_l3_pub  = self.create_publisher(Float64, '/joint_arm_l3', 10)
        self.joint_arm_l2_pub  = self.create_publisher(Float64, '/joint_arm_l2', 10)
        self.joint_arm_l1_pub  = self.create_publisher(Float64, '/joint_arm_l1', 10)
        self.joint_arm_l0_pub  = self.create_publisher(Float64, '/joint_arm_l0', 10)

        self.gripper_left_pub  = self.create_publisher(Float64, '/joint_gripper_finger_left', 10)
        self.gripper_right_pub = self.create_publisher(Float64, '/joint_gripper_finger_right', 10)

        # Publish "True" on /joint_status if we detect two open palms
        self.joint_status_pub  = self.create_publisher(Bool, '/joint_status', 10)

        self.active_joint = None
        self.get_logger().info("[ArmFingerNode] Started, waiting for '/selected_joint'...")

        # We'll store the last 5 raw classifications in a deque (if you still want smoothing)
        self.arm_class_history = deque(maxlen=5)

        # Camera does NOT start until we receive a /selected_joint
        self.camera_started = False
        self.cap = None

        # For controlling how often we publish repeated string messages
        self.prev_message = None
        self.last_publish_time = time.time()
        self.publish_interval = 1.0

        # Initialize Mediapipe solutions (Pose + Hands)
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # NOTE: set max_num_hands=2 so we can detect two hands
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Create a timer to process frames if/once the camera is open (~20 FPS)
        self.timer_ = self.create_timer(0.05, self.timer_callback)

    # =========== Subscriptions ===========
    def joint_callback(self, msg):
        self.active_joint = msg.data
        self.get_logger().info(f"[ArmFingerNode] Received /selected_joint => {self.active_joint}")

        # If camera hasn't started or was previously stopped by two open palms, re-start it
        if not self.camera_started:
            self.start_camera()

    def gesture_callback(self, msg):
        self.get_logger().info(f"[ArmFingerNode] (Listener) Received gesture: {msg.data}")

    # =========== Start/Stop camera ===========
    def start_camera(self):
        """Open webcam feed after first /selected_joint message or after two open palms were shown."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("[ArmFingerNode] Cannot open webcam. No camera feed.")
            return

        self.camera_started = True
        self.get_logger().info("[ArmFingerNode] Camera feed started!")

    def stop_camera(self):
        """Release camera feed and close the OpenCV window."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.camera_started = False
        self.get_logger().info("[ArmFingerNode] Camera feed and window closed.")

    # =========== Timer Callback: process camera if started ===========
    def timer_callback(self):
        """
        Periodic timer. Only processes frames if the camera is open.
        If we detect TWO open palms, we properly close the OpenCV window,
        THEN publish /joint_status => True, and wait for new /selected_joint.
        """
        if not self.camera_started:
            return  # Wait until camera is started

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("[ArmFingerNode] Failed to read frame from camera.")
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert BGR -> RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = self.mp_pose.process(rgb_frame)
        hands_result = self.mp_hands.process(rgb_frame)

        # =========== 0) Check if user is showing TWO OPEN PALMS to "quit" camera ===========
        if hands_result.multi_hand_landmarks and len(hands_result.multi_hand_landmarks) >= 2:
            hand1 = hands_result.multi_hand_landmarks[0]
            hand2 = hands_result.multi_hand_landmarks[1]
            if is_open_palm(hand1, w, h, threshold=60) and is_open_palm(hand2, w, h, threshold=60):
                self.get_logger().info("[ArmFingerNode] TWO open palms => shutting down camera feed & window.")
                self.stop_camera()
                status_msg = Bool(data=True)
                self.joint_status_pub.publish(status_msg)
                self.get_logger().info("[ArmFingerNode] Published True on /joint_status.")
                return  # skip further processing

        # Keep track of a debug message we might publish
        message_to_publish = None

        # =========== 1) BASE JOINT ===========
        if self.active_joint == "base":
            index_fingertip = None
            if hands_result.multi_hand_landmarks:
                # We'll just use the first detected hand for the base
                first_hand = hands_result.multi_hand_landmarks[0]
                index_fingertip = first_hand.landmark[8]

            # compute normalized coords for base
            norm_x, norm_y = get_base_movement(index_fingertip, w, h)

            # publish to /cmd_vel
            twist_msg = Twist()
            twist_msg.linear.x = float(norm_y)
            twist_msg.angular.z = float(norm_x)
            self.cmd_vel_pub.publish(twist_msg)

            self.get_logger().info(
                f"[ArmFingerNode] Publishing Twist => linear.x={norm_y:.2f}, angular.z={norm_x:.2f}"
            )
            message_to_publish = f"moving x={norm_x:.2f}, y={norm_y:.2f}"

        # =========== 2) ARM UP/DOWN ===========
        elif self.active_joint == "arm_up_down":
            if pose_result.pose_landmarks:
                val = self.get_arm_updown_value(pose_result.pose_landmarks.landmark, w, h)
                if val is not None:
                    # Publish this 0..1 value (in steps of 0.1) to /joint_lift
                    self.get_logger().info(f"Arm Up/Down => /joint_lift => {val:.1f}")
                    msg_f = Float64(data=val)
                    self.joint_lift_pub.publish(msg_f)
                    message_to_publish = f"arm_up_down => {val:.1f}"

        # =========== 3) ARM STRETCH/RETRACT ===========
        elif self.active_joint == "arm_stretch_retract":
            if pose_result.pose_landmarks:
                val = self.get_arm_extension_value(pose_result.pose_landmarks.landmark, w, h)
                if val is not None:
                    # Publish this 0..1 value (in steps of 0.1) to all four topics
                    self.get_logger().info(f"Arm Stretch/Retract => /joint_arm_l3.. => {val:.1f}")
                    msg_f = Float64(data=val)
                    self.joint_arm_l3_pub.publish(msg_f)
                    self.joint_arm_l2_pub.publish(msg_f)
                    self.joint_arm_l1_pub.publish(msg_f)
                    self.joint_arm_l0_pub.publish(msg_f)

                    message_to_publish = f"arm_stretch_retract => {val:.1f}"

        # =========== 4) GRIPPER ===========
        elif self.active_joint == "gripper":
            if hands_result.multi_hand_landmarks:
                first_hand = hands_result.multi_hand_landmarks[0]
                gripper_state = get_gripper_state(first_hand, w, h, threshold=50)

                if gripper_state == "gripper close":
                    self.get_logger().info("Gripper Close => left=0.0, right=0.0")
                    msg_left  = Float64(data=0.0)
                    msg_right = Float64(data=0.0)
                    self.gripper_left_pub.publish(msg_left)
                    self.gripper_right_pub.publish(msg_right)

                elif gripper_state == "gripper open":
                    self.get_logger().info("Gripper Open => left=0.9, right=0.9")
                    msg_left  = Float64(data=0.9)
                    msg_right = Float64(data=0.9)
                    self.gripper_left_pub.publish(msg_left)
                    self.gripper_right_pub.publish(msg_right)

                message_to_publish = gripper_state

        # =========== 5) UNKNOWN SELECTION ===========
        else:
            if self.active_joint is None:
                self.get_logger().info("[ArmFingerNode] No joint selected yet.")
            else:
                self.get_logger().info(f"[ArmFingerNode] Unknown joint selection: {self.active_joint}")

        # (Optional) Draw pose/hands on the frame
        if pose_result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                pose_result.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS
            )
        if hands_result.multi_hand_landmarks:
            for hlms in hands_result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hlms, mp.solutions.hands.HAND_CONNECTIONS
                )

        # Publish a textual "gesture" message if it changed from last time, with a 1s interval
        current_time = time.time()
        if message_to_publish is not None:
            if (message_to_publish != self.prev_message) and ((current_time - self.last_publish_time) > self.publish_interval):
                self.publish_gesture(message_to_publish)
                self.prev_message = message_to_publish
                self.last_publish_time = current_time

        # Show the camera feed
        cv2.imshow("[ArmFingerNode] Camera Feed", frame)

        # If user presses 'q' on keyboard, shut down the entire node
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("[ArmFingerNode] 'q' pressed. Shutting down node completely.")
            rclpy.shutdown()
            cv2.destroyAllWindows()

    # ------------- New Helper: Arm Up/Down => 0..1 in 0.1 increments -------------
    def get_arm_updown_value(self, pose_landmarks, w, h):
        """
        Map the vertical difference (shoulder.y - wrist.y) from -200..+200 => 0..1,
        rounding to increments of 0.1. Returns None if landmarks are not sufficiently visible.
        """
        SHOULDER = 12
        WRIST = 16
        if (pose_landmarks[SHOULDER].visibility < 0.6 or
            pose_landmarks[WRIST].visibility < 0.6):
            return None

        s_y = int(pose_landmarks[SHOULDER].y * h)
        w_y = int(pose_landmarks[WRIST].y * h)

        vertical_diff = s_y - w_y  # positive => wrist is below shoulder => "arm up"
        # Clamp to [-200..200]
        vertical_diff = max(-200, min(200, vertical_diff))

        # Map [-200..200] => [0..1]
        #   -200 => 0.0
        #   +200 => 1.0
        val = (vertical_diff + 200) / 400.0  # range in [0..1]

        # Snap to increments of 0.1
        val = round(val * 10) / 10.0

        # Ensure still in [0..1]
        val = max(0.0, min(1.0, val))
        return val

    # ------------- New Helper: Arm Extension => 0..1 in 0.1 increments -------------
    def get_arm_extension_value(self, pose_landmarks, w, h):
        """
        Computes the elbow angle (shoulder->elbow->wrist) in [0..180].
        Maps that angle to [0..1], rounding to increments of 0.1.
        Returns None if landmarks are not sufficiently visible.
        """
        SHOULDER = 12
        ELBOW = 14
        WRIST = 16

        if (pose_landmarks[SHOULDER].visibility < 0.6 or
            pose_landmarks[ELBOW].visibility    < 0.6 or
            pose_landmarks[WRIST].visibility    < 0.6):
            return None

        s_x, s_y = int(pose_landmarks[SHOULDER].x * w), int(pose_landmarks[SHOULDER].y * h)
        e_x, e_y = int(pose_landmarks[ELBOW].x * w),    int(pose_landmarks[ELBOW].y * h)
        w_x, w_y = int(pose_landmarks[WRIST].x * w),    int(pose_landmarks[WRIST].y * h)

        def length(v):
            return math.sqrt(v[0]**2 + v[1]**2)
        def dot(a, b):
            return a[0]*b[0] + a[1]*b[1]

        upper_arm = (e_x - s_x, e_y - s_y)
        lower_arm = (w_x - e_x, w_y - e_y)
        len_u = length(upper_arm)
        len_l = length(lower_arm)
        if len_u < 1e-5 or len_l < 1e-5:
            return None

        cos_angle = dot(upper_arm, lower_arm) / (len_u * len_l)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = math.degrees(math.acos(cos_angle))  # 0..180

        # Map 0..180 => 0..1
        val = angle_deg / 180.0

        # Snap to increments of 0.1
        val = round(val * 10) / 10.0

        # Ensure in [0..1]
        val = max(0.0, min(1.0, val))
        return val

    # =========== Classification / Smoothing Helpers (Optional) ===========
    def get_smoothed_arm_class(self, raw_class):
        """
        If you still want discrete classification (unused now),
        this picks the majority classification in the last 5 frames.
        """
        self.arm_class_history.append(raw_class)
        freq = {}
        for c in self.arm_class_history:
            if c is not None:
                freq[c] = freq.get(c, 0) + 1
        if not freq:
            return None
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        top_class, top_count = sorted_freq[0]
        if top_count < 2:
            return None
        return top_class

    # =========== Publish gesture strings for debug ===========
    def publish_gesture(self, gesture_str):
        msg = String()
        msg.data = gesture_str
        self.gesture_pub.publish(msg)
        self.get_logger().info(f"[ArmFingerNode] (Publisher) Published gesture: {gesture_str}")

    # =========== Cleanup ===========
    def destroy_node(self):
        """Override to ensure we release camera & close windows."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


# =========== Detect an OPEN PALM ===========
def is_open_palm(hand_landmarks, w, h, threshold=60):
    """
    Returns True if all five fingertips (#4, #8, #12, #16, #20)
    are at least 'threshold' pixels away from the palm base (#0).
    """
    if not hand_landmarks:
        return False

    # Palm base is landmark #0
    palm_x = int(hand_landmarks.landmark[0].x * w)
    palm_y = int(hand_landmarks.landmark[0].y * h)

    fingertip_ids = [4, 8, 12, 16, 20]  # thumb tip + index, middle, ring, pinky tips
    for fid in fingertip_ids:
        tip_x = int(hand_landmarks.landmark[fid].x * w)
        tip_y = int(hand_landmarks.landmark[fid].y * h)
        dist = math.sqrt((tip_x - palm_x)**2 + (tip_y - palm_y)**2)
        if dist < threshold:
            # At least one fingertip is too close => not "fully open"
            return False

    return True


# =========== Helpers for "base" and "gripper" movements ===========
def get_base_movement(index_fingertip, w, h, dead_zone=50):
    """
    Returns (norm_x, norm_y) in [-1..1], or (0,0) if finger not found / in dead zone.
    Moves 'base' by interpreting the index fingertip position relative to the center.
    """
    if index_fingertip is None:
        return 0.0, 0.0

    fx = int(index_fingertip.x * w)
    fy = int(index_fingertip.y * h)
    center_x = w // 2
    center_y = h // 2

    dx = fx - center_x
    dy = fy - center_y
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < dead_zone:
        return 0.0, 0.0

    # Normalize to [-1..1]
    norm_x = (fx - center_x) / (w / 2)
    norm_y = (center_y - fy) / (h / 2)
    norm_x = max(-1.0, min(1.0, norm_x))
    norm_y = max(-1.0, min(1.0, norm_y))
    return norm_x, norm_y


def get_gripper_state(hand_landmarks, w, h, threshold=50):
    """
    Simple approach: distance between thumb tip (#4) & index tip (#8).
    If distance < threshold => "gripper close", else => "gripper open".
    """
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    t_x, t_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
    i_x, i_y = int(index_tip.x * w), int(index_tip.y * h)

    dist = math.sqrt((t_x - i_x)**2 + (t_y - i_y)**2)
    if dist < threshold:
        return "gripper close"
    else:
        return "gripper open"


def main():
    rclpy.init()
    node = ArmFingerNode()

    # Spin forever, waiting for either:
    #   1) First /selected_joint to open camera
    #   2) TWO open palms => stop camera feed, publish /joint_status
    #   3) Another /selected_joint => re-start camera
    #   4) 'q' => full shutdown
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    print("[ArmFingerNode] Node shutdown complete.")


if __name__ == "__main__":
    main()
