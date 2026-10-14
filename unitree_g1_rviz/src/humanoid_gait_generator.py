#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
import math

class HumanoidGaitGenerator(Node):
    def __init__(self):
        super().__init__('humanoid_gait_generator')
        
        # Publishers and Broadcasters
        self.br = TransformBroadcaster(self)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.cmd_sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_cb, 10)
        
        # State Variables
        self.x, self.y, self.th = 0.0, 0.0, 0.0  # Position in odom frame
        self.vx, self.vth = 0.0, 0.0            # Commanded velocities
        self.phase = 0.0                        # Animation phase
        
        # Timer for 50Hz update (smooth movement)
        self.timer = self.create_timer(0.02, self.update_logic)

    def cmd_cb(self, msg):
        """Update velocity commands from teleop"""
        self.vx = msg.linear.x
        self.vth = msg.angular.z

    def update_logic(self):
        now = self.get_clock().now().to_msg()
        dt = 0.02

        # --- 1. ODOMETRY LOGIC ---
        self.th += self.vth * dt
        self.x += self.vx * math.cos(self.th) * dt
        self.y += self.vx * math.sin(self.th) * dt

        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'odom'
        t.child_frame_id = 'pelvis'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        # Dynamic height: slightly bob up and down while walking
        bobbing = abs(math.cos(self.phase * 2.0)) * 0.02 if abs(self.vx) > 0.01 else 0.0
        t.transform.translation.z = 0.78 - bobbing 
        
        t.transform.rotation.z = math.sin(self.th / 2.0)
        t.transform.rotation.w = math.cos(self.th / 2.0)
        self.br.sendTransform(t)

        # --- 2. GAIT LOGIC ---
        js = JointState()
        js.header.stamp = now
        
        # List EVERY joint in your URDF to prevent flickering
        js.name = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
            'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
        ]
        
        # Initialize all 29 joints to 0.0
        js.position = [0.0] * len(js.name)

        if abs(self.vx) > 0.01 or abs(self.vth) > 0.01:
            self.phase += (abs(self.vx) * 10.0 + 2.0) * dt
            s = math.sin(self.phase)
            c = math.cos(self.phase)
            
            # Update specific indices for walking (Mapping based on list above)
            js.position[0] = s * 0.45      # left_hip_pitch
            js.position[6] = -s * 0.45     # right_hip_pitch
            js.position[3] = (math.sin(self.phase - 1.5) + 1.1) * 0.5 # left_knee
            js.position[9] = (math.sin(self.phase + 1.5) + 1.1) * 0.5 # right_knee
            js.position[15] = -s * 0.5     # left_shoulder_pitch
            js.position[22] = s * 0.5      # right_shoulder_pitch
            js.position[18] = 0.4          # left_elbow
            js.position[25] = 0.4          # right_elbow
        else:
            # Smoothly decay the phase so it doesn't snap
            self.phase *= 0.9
            if self.phase < 0.01: self.phase = 0.0

        self.joint_pub.publish(js)

    def get_target_pose(self):
        """
        Calculates the current joint positions based on the internal phase
        and returns them as a list of 29 angles for MuJoCo control.
        """
        # Initialize the 29 joint positions
        pos = [0.0] * 29
        
        # Apply the same walking logic used in your JointState publisher
        if abs(self.vx) > 0.01 or abs(self.vth) > 0.01:
            s = math.sin(self.phase)
            
            # Map values exactly as defined in your js.name list
            pos[0] = s * 0.45      # left_hip_pitch
            pos[6] = -s * 0.45     # right_hip_pitch
            pos[3] = (math.sin(self.phase - 1.5) + 1.1) * 0.5 # left_knee
            pos[9] = (math.sin(self.phase + 1.5) + 1.1) * 0.5 # right_knee
            pos[15] = -s * 0.5     # left_shoulder_pitch
            pos[22] = s * 0.5      # right_shoulder_pitch
            pos[18] = 0.4          # left_elbow
            pos[25] = 0.4          # right_elbow
            
        return pos

def main(args=None):

    rclpy.init(args=args)
    node = HumanoidGaitGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()