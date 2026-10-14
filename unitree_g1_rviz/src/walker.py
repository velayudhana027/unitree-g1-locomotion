#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import TransformBroadcaster
import math

class VirtualG1Walker(Node):
    def __init__(self):
        super().__init__('virtual_g1_walker')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.subscription = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        
        # Robot Position State
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.vx = 0.0
        self.vth = 0.0
        
        self.timer = self.create_timer(0.05, self.update_and_publish) # 20Hz update

    def cmd_vel_callback(self, msg):
        self.vx = msg.linear.x
        self.vth = msg.angular.z

    def update_and_publish(self):
        dt = 0.05
        # Update pose based on velocity
        self.th += self.vth * dt
        self.x += self.vx * math.cos(self.th) * dt
        self.y += self.vx * math.sin(self.th) * dt

        # Create and send the transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'pelvis'

        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.7  # Standing height for G1

        # Quaternion from yaw
        t.transform.rotation.z = math.sin(self.th / 2.0)
        t.transform.rotation.w = math.cos(self.th / 2.0)

        self.tf_broadcaster.sendTransform(t)

def main():
    rclpy.init()
    node = VirtualG1Walker()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()