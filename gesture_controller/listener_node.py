#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ListenerNode(Node):
    """
    Subscribes to:
      - '/selected_joint' (which joint user selected from Menu)
      - '/gesture_commands' (arm/finger states)

    Just prints them out. 
    In a real system, you could command the robot base or arm here.
    """

    def __init__(self):
        super().__init__('listener_node')

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

        self.get_logger().info("Listener Node started. Subscribing to '/selected_joint' and '/gesture_commands'.")

    def joint_callback(self, msg):
        self.get_logger().info(f"[Listener] Selected joint: {msg.data}")

    def gesture_callback(self, msg):
        self.get_logger().info(f"[Listener] Gesture: {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = ListenerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
