#!/usr/bin/env python3
import subprocess
import sys
import time
import rclpy
from rclpy.node import Node
from lifecycle_msgs.msg import Transition

class UISupervisor(Node):
    def __init__(self):
        super().__init__('ui_supervisor')
        self.current_process = None
        self.start_menu()

    def start_menu(self):
        """Launch robot_menu.py as a subprocess"""
        self.get_logger().info("Starting robot menu...")
        self.current_process = subprocess.Popen(
            [sys.executable, "robot_menu.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        self.create_monitor_timer()

    def start_hand_tracking(self):
        """Launch HandTrackingNode as a subprocess"""
        self.get_logger().info("Starting hand tracking UI...")
        self.current_process = subprocess.Popen(
            [sys.executable, "ros_pointing_ui.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        self.create_monitor_timer()

    def create_monitor_timer(self):
        """Create timer to check process status"""
        self.monitor_timer = self.create_timer(1.0, self.monitor_process)

    def monitor_process(self):
        """Monitor current subprocess and handle transitions"""
        if self.current_process.poll() is not None:
            self.monitor_timer.cancel()
            return_code = self.current_process.returncode
            
            if return_code == 0:  # Normal exit
                if "ros_pointing_ui.py" in " ".join(self.current_process.args):
                    self.get_logger().info("Hand tracking exited, restarting menu...")
                    self.start_menu()
                else:
                    self.get_logger().info("Menu exited, starting hand tracking...")
                    self.start_hand_tracking()
            else:  # Error handling
                self.get_logger().error(
                    f"Process crashed with code {return_code}. Restarting menu..."
                )
                self.start_menu()

    def shutdown(self):
        """Cleanup resources"""
        if self.current_process:
            self.current_process.terminate()
            self.current_process.wait()
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    supervisor = UISupervisor()
    
    try:
        rclpy.spin(supervisor)
    except KeyboardInterrupt:
        supervisor.get_logger().info("Shutting down supervisor...")
    finally:
        supervisor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
