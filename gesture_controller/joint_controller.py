import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist


class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')
        
        # Initialize variables to store subscribed data
        self.joint_topic = ""
        self.point_data = Point()

        # Create subscribers
        self.string_subscriber = self.create_subscription(
            String,
            '/selected_joint',
            self.string_callback,
            10
        )
        self.point_subscriber = self.create_subscription(
            Point,
            '/hand_tracking/position',
            self.point_callback,
            10
        )

        # Create publishers
        self.publisher_1 = self.create_publisher(Twist, '/cmd_vel', 10)
        self.publisher_2 = self.create_publisher(String, 'joint_2_topic', 10)
        self.publisher_3 = self.create_publisher(String, 'joint_3_topic', 10)
        self.publisher_4 = self.create_publisher(String, 'joint_4_topic', 10)


    def string_callback(self, msg):
        self.joint_topic = msg.data
        

    def point_callback(self, msg):
        self.point_data = msg
        self.publish_message()

    def publish_message(self):
        msg = String()
        if self.joint_topic == "joint1":
            twist_msg = Twist()
            twist_msg.linear.x = 0.5*self.point_data.x
            twist_msg.angular.z = 0.5* self.point_data.y
            
            self.publisher_1.publish(twist_msg)
        elif self.joint_topic == "joint_2":
            msg.data = f"Joint 2 activated with point: x={self.point_data.x}, y={self.point_data.y}, z={self.point_data.z}"
            self.publisher_2.publish(msg)
        elif self.joint_topic == "joint_3":
            msg.data = f"Joint 3 activated with point: x={self.point_data.x}, y={self.point_data.y}, z={self.point_data.z}"
            self.publisher_3.publish(msg)
        elif self.joint_topic == "joint_4":
            msg.data = f"Joint 4 activated with point: x={self.point_data.x}, y={self.point_data.y}, z={self.point_data.z}"
            self.publisher_4.publish(msg)
        else:
            self.get_logger().info("No valid joint_topic specified for publishing.")

def main(args=None):
    rclpy.init(args=args)
    node = JointController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()