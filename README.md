# Gesture-Based Control Interface for the Stretch Robot

## Authors

- Atharva Jamsandekar ([AtharvaJ11](https://github.com/AtharvaJ11))  
- Shubhankar Katekari ([ShubhankarKatekari](https://github.com/ShubhankarKatekari))  
- Nikhil Gutlapalli ([GutlapalliNikhil](https://github.com/GutlapalliNikhil))  


## Project Description

Controlling mobile manipulators like the Stretch robot typically involves keyboard or joystick-based teleoperation, which can be unintuitive or inefficient in real-world human-robot interaction. Our project proposes a gesture-based control interface that allows users to intuitively command both the mobile base and the manipulator arm of the Stretch robot using hand gestures. These gestures will be interpreted from a video feed and translated into control commands for movement and object manipulation within the Gazebo simulation environment. The goal is to enable more natural, user-friendly robot control.

## Demo and Presentation

- **Presentation Video:** [Presentation](https://drive.google.com/file/d/1FOg8Z7usrDypcE-0spp5fT6KVNBRtJWK/view)  
- **Demo Video(s):** [Demonstration](https://drive.google.com/file/d/1Y-RRnm93idmmsbtF1hLAsq5z70bYDb3g/view?usp=drive_link)  


## Setup Instructions

1. **Install ROS 2**: Ensure that ROS 2 (Humble or later) is installed on your system. Follow the official [ROS 2 installation guide](https://docs.ros.org/en/rolling/Installation.html) if needed.

2. **Clone the Repository**: Navigate to your workspace and clone the repository:
    ```bash
    git clone https://github.com/AtharvaJ11/gesture-based-teleop.git
    ```

3. **Install Dependencies**: Install the required Python dependencies:
    ```bash
    rosdep install --from-paths src --ignore-src -r -y
    ```

4. **Build the Workspace**: Build the ROS 2 workspace:
    ```bash
    colcon build
    ```

5. **Source the Workspace**: Source the setup file to overlay the workspace:
    ```bash
    source install/setup.bash
    ```

## Running Instructions

1. **Run `arm_finger_node`**: Open a second terminal and run the `arm_finger_node` script using Python:
    ```bash
    python3 ~/CS5330/finalProject/ros2_ws/src/gesture_controller/arm_finger_node.py
    ```

2. **Run `menu_node`**: Open a third terminal and run the `menu_node` script using Python:
    ```bash
    python3 ~/CS5330/finalProject/ros2_ws/src/gesture_controller/menu_node.py
    ```
