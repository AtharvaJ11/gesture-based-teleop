#!/usr/bin/env python3
"""
menu_node.py
Authors: Atharva Jamsandekar, Shubhankar Katekari, and Nikhil Gutlapalli
Date: 04-23-2025

Description:
This script implements a hover-based menu system using OpenCV and Mediapipe for gesture detection,
integrated with ROS 2 for publishing selected menu options. The menu allows users to select joints
or a "continue" option by hovering their fingertip over buttons displayed on the screen. The selection
is confirmed after a specified hover duration. The selected joint is published to a ROS 2 topic
('/selected_joint') **only after** the "continue" option is selected.

Additional Feature:
- Subscribes to /joint_status (Bool). If it receives `True`, it *restarts* the webcam menu loop.
"""

import cv2
import mediapipe as mp
import time

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool


class JointPublisherNode(Node):
    def __init__(self):
        super().__init__('joint_selector_node')
        self.publisher_ = self.create_publisher(String, '/selected_joint', 10)

        # NEW: subscribe to /joint_status
        self.status_sub = self.create_subscription(
            Bool,
            '/joint_status',
            self.joint_status_callback,
            10
        )

        # Internal flags
        self.selected_joint = None
        self.restart_menu_flag = False   # If True => main() will re-run the menu

        self.get_logger().info("[MenuNode] Started. Waiting for final selection.")

    def publish_joint_once(self, joint_name: str):
        msg = String()
        msg.data = joint_name
        self.publisher_.publish(msg)
        self.get_logger().info(f"[MenuNode] Published selected joint: {joint_name}")

    def joint_status_callback(self, msg: Bool):
        """
        If we receive True on /joint_status, we set restart_menu_flag=True.
        main() sees this flag and re-launches the menu.
        """
        if msg.data is True:
            self.get_logger().info("[MenuNode] /joint_status => True => Restarting menu on next spin.")
            self.restart_menu_flag = True


# -----------------------
# Layout & Constants
# -----------------------
joint_buttons = {
    "base":                (50,  50,  300, 100),
    "arm_up_down":         (50,  110, 300, 160),
    "arm_stretch_retract": (50,  170, 300, 220),
    "gripper":             (50,  230, 300, 280),
}

continue_box = ("continue", (50, 290, 300, 340))

HOVER_TIME = 2.0

# Each entry has a None time initially
hover_start_time = {name: None for name in joint_buttons}
hover_start_time["continue"] = None

COLOR_IDLE     = (200, 200, 200)
COLOR_HOVER    = (50, 200, 255)
COLOR_SELECTED = (0, 255, 0)


def is_point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return (x1 <= px <= x2) and (y1 <= py <= y2)


def draw_button(frame, text, box, state="idle"):
    (x1, y1, x2, y2) = box
    if state == "hover":
        color = COLOR_HOVER
    elif state == "selected":
        color = COLOR_SELECTED
    else:
        color = COLOR_IDLE

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness  = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    center_x = x1 + (x2 - x1)//2
    center_y = y1 + (y2 - y1)//2

    text_x = center_x - text_w//2
    text_y = center_y + text_h//2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)


def run_menu(node: JointPublisherNode):
    """
    Runs one pass of the menu loop:
      - Opens webcam
      - Waits for user to hover over joints / continue
      - Once "continue" is selected, returns
      - Closes webcam window
      - Publishes selected joint if any
    """
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[MenuNode] Error: Cannot open webcam.")
        return

    # Reset any previous selection and hover times
    node.selected_joint = None
    for k in hover_start_time:
        hover_start_time[k] = None

    continue_chosen = False

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            # Let ROS process any incoming callbacks
            rclpy.spin_once(node, timeout_sec=0.01)

            ret, frame = cap.read()
            if not ret:
                print("[MenuNode] Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            fingertip_x, fingertip_y = None, None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                index_tip = hand_landmarks.landmark[8]
                fingertip_x = int(index_tip.x * w)
                fingertip_y = int(index_tip.y * h)
                cv2.circle(frame, (fingertip_x, fingertip_y), 10, (0, 0, 255), -1)

            # --- Draw & handle joint buttons ---
            for btn_name, box in joint_buttons.items():
                if fingertip_x is not None and fingertip_y is not None:
                    inside = is_point_in_box(fingertip_x, fingertip_y, box)
                else:
                    inside = False

                if inside:
                    if hover_start_time[btn_name] is None:
                        hover_start_time[btn_name] = time.time()
                    else:
                        elapsed = time.time() - hover_start_time[btn_name]
                        if elapsed >= HOVER_TIME:
                            # Toggle selection
                            if node.selected_joint is None:
                                node.selected_joint = btn_name
                                print(f"[MenuNode] => SELECTED {btn_name}")
                            elif node.selected_joint == btn_name:
                                # Deselect
                                node.selected_joint = None
                                print(f"[MenuNode] => DESELECTED {btn_name}")
                            hover_start_time[btn_name] = None

                    # Display
                    if node.selected_joint == btn_name:
                        draw_button(frame, btn_name, box, state="selected")
                    else:
                        # Still hovering, but not yet triggered
                        elapsed = time.time() - hover_start_time[btn_name]
                        if elapsed < HOVER_TIME:
                            draw_button(frame, btn_name, box, state="hover")
                        else:
                            draw_button(frame, btn_name, box, state="idle")
                else:
                    hover_start_time[btn_name] = None
                    if node.selected_joint == btn_name:
                        draw_button(frame, btn_name, box, state="selected")
                    else:
                        draw_button(frame, btn_name, box, state="idle")

            # --- Handle "Continue" button ---
            c_name, c_box = continue_box
            if fingertip_x is not None and fingertip_y is not None:
                inside_continue = is_point_in_box(fingertip_x, fingertip_y, c_box)
            else:
                inside_continue = False

            if inside_continue:
                if hover_start_time["continue"] is None:
                    hover_start_time["continue"] = time.time()
                else:
                    elapsed = time.time() - hover_start_time["continue"]
                    if elapsed >= HOVER_TIME:
                        print("[MenuNode] 'Continue' selected. Exiting menu loop.")
                        continue_chosen = True
                        break
                draw_button(frame, c_name, c_box, state="hover")
            else:
                hover_start_time["continue"] = None
                draw_button(frame, c_name, c_box, state="idle")

            cv2.imshow("[MenuNode] Hover-based Menu", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[MenuNode] 'q' pressed. Exiting menu loop.")
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # After exiting the loop, if user selected "continue"
    if continue_chosen:
        if node.selected_joint:
            node.publish_joint_once(node.selected_joint)
            print(f"[MenuNode] => Published final selection: {node.selected_joint}")
        else:
            print("[MenuNode] => No joint selected. Nothing to publish.")


def main():
    rclpy.init()
    node = JointPublisherNode()

    print("[MenuNode] Running the menu for the first time.")
    run_menu(node)

    # Now keep the node alive, listening to /joint_status.
    # If /joint_status => True, we run_menu(node) again.
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.restart_menu_flag:
                node.restart_menu_flag = False
                print("[MenuNode] Restarting the menu due to /joint_status => True.")
                run_menu(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    print("[MenuNode] Done.")


if __name__ == "__main__":
    main()
