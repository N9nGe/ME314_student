#!/usr/bin/env python3
import cv2
import numpy as np
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PointStamped
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from builtin_interfaces.msg import Time

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

        self.last_red_pixel = None
        self.fx = 640.5098266601562
        self.fy = 640.5098266601562
        self.cx = 640.0
        self.cy = 360.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.target_pose = None
        self.object_locked = False
        self.origin_pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # Default origin pose

    def arm_pose_callback(self, msg: Pose):
        self.current_arm_pose = msg

    def gripper_position_callback(self, msg: Float64):
        self.current_gripper_position = msg.data

    def color_image_callback(self, msg: Image):
        if self.object_locked:
            return
        # encoding = rgb8
        
        raw_image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        # Save the image for testing
        # cv2.imwrite('image_rgb.png', rgb_image)

        # get the center of the red box
        center = self.red_box_segmentation(raw_image)
        if center is not None:
            cx, cy = center
            self.get_logger().info(f"Red object center: ({cx}, {cy})")

            # Test code
            # rgb_image_marked = rgb_image.copy()
            # cv2.circle(rgb_image_marked, (cx, cy), 8, (0, 255, 0), thickness=2)  # 绿色圆圈
            # cv2.imwrite('rgb_image_marked.png', rgb_image_marked)

            # store the current coordinate
            self.last_red_pixel = (cx, cy)
        else:
            self.get_logger().info("Red object not found.")


    def depth_image_callback(self, msg: Image):
        if self.object_locked or self.last_red_pixel is None:
            return
        depth_image_raw = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        # for display 
        depth_image = cv2.normalize(depth_image_raw, None, 0, 255, cv2.NORM_MINMAX)
        depth_image = depth_image.astype(np.uint8)
        # cv2.imwrite('depth_display.png', depth_image)
        if hasattr(self, 'last_red_pixel'):
            u, v = self.last_red_pixel
            z_mm = depth_image_raw[v, u] 

            if z_mm == 0:
                self.get_logger().warn("Invalid depth at red object pixel.")
                return

            x, y, z = self.pixel_to_camera_point(u, v, z_mm)
            point_cam = [x, y, z]
            point_world = self.transform_camera_to_world_tf(point_cam)

            if point_world is not None:
                self.target_pose = [point_world[0], point_world[1], point_world[2], 1.0, 0.0, 0.0, 0.0]  
                self.object_locked = True 
                self.publish_pose(self.target_pose)
                self.publish_gripper_position(1.0)  # Close the gripper
                time.sleep(2.0)
                self.publish_pose(self.origin_pose)
                

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

    def red_box_segmentation(self, rgb_image):
        # rgb mask
        lower_red = np.array([150, 0, 0])
        upper_red = np.array([255, 100, 100])

        red_mask = cv2.inRange(rgb_image, lower_red, upper_red)
        cv2.imwrite("red_mask_rgb.png", red_mask)
        # find the largest contour
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None  # no red box found
        largest_contour = max(contours, key=cv2.contourArea)

        # get the center of mass
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    
    def pixel_to_camera_point(self, u, v, z_mm):
        z_m = z_mm / 1000.0  # mm -> m
        x = (u - self.cx) * z_m / self.fx
        y = (v - self.cy) * z_m / self.fy
        return (x, y, z_m)
    
    def transform_camera_to_world_tf(self, point_cam, frame='camera_depth_optical_frame'):
        point_msg = PointStamped()
        point_msg.header.frame_id = frame
        # point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.stamp = Time()
        point_msg.point.x = point_cam[0]
        point_msg.point.y = point_cam[1]
        point_msg.point.z = point_cam[2]

        try:
            point_world = self.tf_buffer.transform(point_msg, 'world', timeout=rclpy.duration.Duration(seconds=0.5))
            return (point_world.point.x, point_world.point.y, point_world.point.z)
        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {str(e)}")
            return None
    
def main(args=None):
    rclpy.init(args=args)
    node = PickPlace()

    # try:
    #     rclpy.spin(node)
    # except KeyboardInterrupt:
    #     pass

    # Define poses using the array format [x, y, z, qx, qy, qz, qw]
    p0 = [0.35442898432566816, 0.08758732426157431, 0.007964988540690887, 1.0, 0.0, 0.0, 0.0]
    p1 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    # poses = [p0, p1]
    node.publish_pose(p0)
    node.publish_gripper_position(1.0)
    node.publish_pose(p1)

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