#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
import cv2
import numpy as np

# Import the command queue message types from the reference code
from me314_msgs.msg import CommandQueue, CommandWrapper

# color image topic color/image_raw

class PickPlace(Node):
    def __init__(self):
        super().__init__('PickPlace')
        
        # Replace the direct publishers with the command queue publisher
        self.command_queue_pub = self.create_publisher(CommandQueue, '/me314_xarm_command_queue', 10)
        
        # Subscribe to current arm pose and gripper position for status tracking (optional)
        self.current_arm_pose = None
        self.pose_status_sub = self.create_subscription(Pose, '/me314_xarm_current_pose', self.arm_pose_callback, 10)
        
        self.current_gripper_position = None
        self.gripper_status_sub = self.create_subscription(Float64, '/me314_xarm_gripper_position', self.gripper_position_callback, 10)

        # image subscriber
        self.rgb_subscription = self.create_subscription(Image,'/color/image_raw', self.color_image_callback, 10)
        self.depth_subscription = self.create_subscription(Image,'/aligned_depth_to_color/image_raw', self.depth_image_callback, 10)

        self.rgb_image = None
        self.depth_image = None

    def arm_pose_callback(self, msg: Pose):
        self.current_arm_pose = msg

    def gripper_position_callback(self, msg: Float64):
        self.current_gripper_position = msg.data

    def color_image_callback(self, msg: Image):
        # encoding = rgb8
        raw_image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        # Save the image for testing
        # cv2.imwrite('image_rgb.png', rgb_image)
        self.rgb_image = rgb_image


    def depth_image_callback(self, msg: Image):
        depth_image = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image = depth_image.astype(np.uint8)
        self.depth_image = depth_image
        # cv2.imwrite('depth_display.png', depth_display)

    def publish_pose(self, pose_array: list):
        """
        Publishes a pose command to the command queue using an array format.
        pose_array format: [x, y, z, qx, qy, qz, qw]
        """
        # Create a CommandQueue message containing a single pose command
        queue_msg = CommandQueue()
        queue_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Create a CommandWrapper for the pose command
        wrapper = CommandWrapper()
        wrapper.command_type = "pose"
        
        # Populate the pose_command with the values from the pose_array
        wrapper.pose_command.x = pose_array[0]
        wrapper.pose_command.y = pose_array[1]
        wrapper.pose_command.z = pose_array[2]
        wrapper.pose_command.qx = pose_array[3]
        wrapper.pose_command.qy = pose_array[4]
        wrapper.pose_command.qz = pose_array[5]
        wrapper.pose_command.qw = pose_array[6]
        
        # Add the command to the queue and publish
        queue_msg.commands.append(wrapper)
        self.command_queue_pub.publish(queue_msg)
        
        self.get_logger().info(f"Published Pose to command queue:\n"
                               f"  position=({pose_array[0]}, {pose_array[1]}, {pose_array[2]})\n"
                               f"  orientation=({pose_array[3]}, {pose_array[4]}, "
                               f"{pose_array[5]}, {pose_array[6]})")

    def publish_gripper_position(self, gripper_pos: float):
        """
        Publishes a gripper command to the command queue.
        For example:
          0.0 is "fully open"
          1.0 is "closed"
        """
        # Create a CommandQueue message containing a single gripper command
        queue_msg = CommandQueue()
        queue_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Create a CommandWrapper for the gripper command
        wrapper = CommandWrapper()
        wrapper.command_type = "gripper"
        wrapper.gripper_command.gripper_position = gripper_pos
        
        # Add the command to the queue and publish
        queue_msg.commands.append(wrapper)
        self.command_queue_pub.publish(queue_msg)
        
        self.get_logger().info(f"Published gripper command to queue: {gripper_pos:.2f}")


def main(args=None):
    rclpy.init(args=args)
    node = PickPlace()
    
    #TODO: test the color image first
    node.get_logger().info("Opening gripper...")
    node.publish_gripper_position(0.0)
    p0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # TODO: Test the start pose of the robot arm
    # node.get_logger().info(f"Publishing Pose {i+1}...")
    node.publish_pose(p0)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # # Define poses using the array format [x, y, z, qx, qy, qz, qw]
    # p0 = [0.3408, 0.0021, 0.3029, 1.0, 0.0, 0.0, 0.0]
    # p1 = [p0[0], p0[1], 0.1, 1.0, 0.0, 0.0, 0.0]

    # poses = [p0, p1]

    # # Let's first open the gripper (0.0 to 1.0, where 0.0 is fully open and 1.0 is fully closed)
    # node.get_logger().info("Opening gripper...")
    # node.publish_gripper_position(0.0)

    # # Move the arm to each pose
    # for i, pose in enumerate(poses):
    #     node.get_logger().info(f"Publishing Pose {i+1}...")
    #     node.publish_pose(pose)

    # # Now close the gripper.
    # node.get_logger().info("Closing gripper...")
    # node.publish_gripper_position(1.0)

    node.get_logger().info("All actions done. Shutting down.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()