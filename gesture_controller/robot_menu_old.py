#!/usr/bin/env python3

import cv2
import mediapipe as mp
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import sys



# -----------------------
# ROS 2 Publisher Node
# -----------------------
class JointPublisherNode(Node):
    """
    Simple ROS 2 node that publishes a std_msgs/String to the '/selected_joint' topic
    whenever a joint is selected.
    """
    def __init__(self):
        super().__init__('joint_selector_node')
        self.publisher_ = self.create_publisher(String, '/selected_joint', 10)

    def publish_joint(self, joint_name: str):
        msg = String()
        msg.data = joint_name
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published selected joint: {joint_name}')


# -----------------------
# Layout & Constants
# -----------------------

# Define bounding boxes for the four joint buttons: (x1, y1, x2, y2)
joint_buttons = {
    "joint1": (50,  50,  300, 100),
    "joint2": (50,  110, 300, 160),
    "joint3": (50,  170, 300, 220),
    "joint4": (50,  230, 300, 280),
}

# Define bounding box for "Continue" button
continue_box = ("continue", (50, 310, 300, 360))

# Hover time in seconds required to select/deselect a button
HOVER_TIME = 3.0

# Hover start times for each button
hover_start_time = {name: None for name in joint_buttons}
hover_start_time["continue"] = None

# Currently selected joint (None means no joint selected)
selected_joint = None

# Colors for drawing
COLOR_IDLE     = (200, 200, 200)   # Gray
COLOR_HOVER    = (50, 200, 255)    # Teal
COLOR_SELECTED = (0, 255, 0)       # Green


# -----------------------
# Helper Functions
# -----------------------

def is_point_in_box(px, py, box):
    """Check if (px, py) lies in the bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    return (x1 <= px <= x2) and (y1 <= py <= y2)

def draw_button(frame, text, box, state="idle"):
    """
    Draw a nicer-looking button with a semi-transparent fill + border + centered text.
    state can be 'idle', 'hover', or 'selected'.
    """
    (x1, y1, x2, y2) = box
    if state == "hover":
        color = COLOR_HOVER
    elif state == "selected":
        color = COLOR_SELECTED
    else:
        color = COLOR_IDLE

    # Draw a filled overlay rectangle for a semi-transparent look
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    alpha = 0.2  # transparency factor
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Solid border
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Center text
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


# -----------------------
# Main Function
# -----------------------

def main():
    # Initialize ROS 2
    rclpy.init()
    node = JointPublisherNode()

    # Initialize MediaPipe for hand detection
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        rclpy.shutdown()
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        global selected_joint

        while True:
            # Let ROS 2 process any callbacks (if needed)
            rclpy.spin_once(node, timeout_sec=0.01)

            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)  # mirror view
            h, w, _ = frame.shape

            # Convert to RGB for Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Track only the index fingertip
            fingertip_x, fingertip_y = None, None
            if results.multi_hand_landmarks:
                # Only consider the first hand
                hand_landmarks = results.multi_hand_landmarks[0]
                # Index fingertip = landmark #8
                index_tip = hand_landmarks.landmark[8]
                fingertip_x = int(index_tip.x * w)
                fingertip_y = int(index_tip.y * h)
                # Draw fingertip as a small circle
                cv2.circle(frame, (fingertip_x, fingertip_y), 10, (0, 0, 255), -1)

            # Check each joint button
            for btn_name, box in joint_buttons.items():
                # Are we hovering this button?
                if fingertip_x is not None and fingertip_y is not None:
                    inside = is_point_in_box(fingertip_x, fingertip_y, box)
                else:
                    inside = False

                # If inside the button region
                if inside:
                    if hover_start_time[btn_name] is None:
                        hover_start_time[btn_name] = time.time()
                    else:
                        elapsed = time.time() - hover_start_time[btn_name]
                        if elapsed >= HOVER_TIME:
                            # 3 seconds have passed
                            if selected_joint is None:
                                # No joint is selected => select this one
                                selected_joint = btn_name
                                node.publish_joint(selected_joint)
                            elif selected_joint == btn_name:
                                # Hovering again on the same joint => deselect
                                selected_joint = None
                            # Either way, reset the hover timer so we don't re-trigger instantly
                            hover_start_time[btn_name] = None
                    # Determine the visual state
                    # If it's already selected, we color it green
                    if selected_joint == btn_name:
                        draw_button(frame, btn_name, box, state="selected")
                    else:
                        # if it's not selected yet, we show hover color
                        elapsed = time.time() - hover_start_time[btn_name]
                        if elapsed < HOVER_TIME:
                            draw_button(frame, btn_name, box, state="hover")
                        else:
                            # If we just triggered the selection/deselection,
                            # it might be selected or none now.
                            if selected_joint == btn_name:
                                draw_button(frame, btn_name, box, state="selected")
                            else:
                                draw_button(frame, btn_name, box, state="idle")
                else:
                    # Not inside the button => reset its hover timer
                    hover_start_time[btn_name] = None
                    # Draw selected joint as green, else idle
                    if selected_joint == btn_name:
                        draw_button(frame, btn_name, box, state="selected")
                    else:
                        draw_button(frame, btn_name, box, state="idle")

            # Check the "Continue" button
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
                        print("Continue button selected. Exiting...")
                        break
                # Show hover style on the continue button
                draw_button(frame, c_name, c_box, state="hover")
            else:
                hover_start_time["continue"] = None
                draw_button(frame, c_name, c_box, state="idle")

            # Show the image
            cv2.imshow("STRETCH Robot Control (Hover-based)", frame)

            # Press 'q' to quit any time
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User pressed 'q'. Exiting...")
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()
    print("ROS 2 node shut down. Program finished.")


if __name__ == "__main__":
    main()
