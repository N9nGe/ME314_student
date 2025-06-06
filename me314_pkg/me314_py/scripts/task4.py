#!/usr/bin/env python3
import cv2
import numpy as np
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PointStamped
from std_msgs.msg import Float64, Bool
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

        self.arm_executing = False
        self.arm_executing_sub = self.create_subscription(Bool, '/me314_xarm_is_executing', self.arm_executing_callback, 10)
        
        self.collision_check = False
        self.collision_sub = self.create_subscription(Bool, '/me314_xarm_collision', self.collision_callback, 10)

        self.queue_size = 0.0
        self.queue_size_sub = self.create_subscription(Float64, '/me314_xarm_queue_size', self.queue_size_callback, 10)

        # image subscriber
        self.rgb_subscription = self.create_subscription(Image,'/camera/realsense2_camera_node/color/image_raw', self.color_image_callback, 10)
        self.depth_subscription = self.create_subscription(Image,'/camera/realsense2_camera_node/aligned_depth_to_color/image_raw', self.depth_image_callback, 10)

        self.last_grey_pixel = None
        self.last_green_pixel = None
        
        self.fx = 908.6455078125
        self.fy = 909.2957153320312
        self.cx = 646.2830810546875
        self.cy = 373.0643615722656

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pick_target_pose = None
        self.place_target_pose = None
        self.object_locked = False
        self.origin_pose = [0.16664958131555577, 0.0028170095361637172, 0.18776892332246256, 1.0, 0.0, 0.0, 0.0]  # Default origin pose

        self.pre_timer = self.create_timer(1.0, self.timer_callback)
        # self.timer = self.create_timer(1.0, self.timer_callback)

        self.task4_done = False
        self.release_flag = False
        

    def timer_callback(self):
        if not self.task4_done:
            if self.pick_target_pose and self.place_target_pose and not self.object_locked:
                self.get_logger().info("Timer triggered pick-and-place.")
                self.pre_coin(self.pick_target_pose, self.place_target_pose)
                self.task4_done = True
                self.object_locked = True

    def arm_pose_callback(self, msg: Pose):
        self.current_arm_pose = msg

    def gripper_position_callback(self, msg: Float64):
        self.current_gripper_position = msg.data

    def arm_executing_callback(self, msg: Bool):
        self.arm_executing = msg.data

    def collision_callback(self, msg: Bool):
        if msg.data == True:
            self.collision_check = True
    
    def queue_size_callback(self, msg: Float64):
        self.queue_size = msg.data

    def color_image_callback(self, msg: Image):
        if self.object_locked or self.arm_executing:
            return
        # encoding = rgb8
        
        raw_image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        # rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        # Save the image for testing
        # cv2.imwrite('image_rgb.png', rgb_image)

        # get the center of the dollor and green box
        dollor_center = self.color_box_segmentation(raw_image, 'yellow')
        if dollor_center is not None:
            cx, cy = dollor_center
            self.get_logger().info(f"grey object center: ({cx}, {cy})")

            # Test code
            rgb_image_marked = raw_image.copy()
            cv2.circle(rgb_image_marked, (cx, cy), 8, (0, 255, 0), thickness=2)  # 绿色圆圈
            cv2.imwrite('grey_image_marked.png', rgb_image_marked)

            # store the current coordinate
            self.last_grey_pixel = (cx, cy)
        else:
            self.get_logger().info("grey object not found.")

        green_center = self.color_box_segmentation(raw_image, 'green')
        if green_center is not None:
            cx, cy = green_center
            self.get_logger().info(f"Green object center: ({cx}, {cy})")

            # Test code
            green_image_marked = raw_image.copy()
            cv2.circle(green_image_marked, (cx, cy), 8, (255, 0, 0), thickness=2)  # 绿色圆圈
            cv2.imwrite('green_image_marked.png', green_image_marked)

            self.last_green_pixel = (cx, cy)
        else:
            self.get_logger().info("Green object not found.")

    def depth_image_callback(self, msg: Image):
        if self.object_locked or self.last_grey_pixel is None or self.arm_executing:
            return
        depth_image_raw = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        # for display 
        depth_image = cv2.normalize(depth_image_raw, None, 0, 255, cv2.NORM_MINMAX)
        depth_image = depth_image.astype(np.uint8)
        # cv2.imwrite('depth_display.png', depth_image)
        if self.last_grey_pixel is not None:
            u, v = self.last_grey_pixel
            z_mm = depth_image_raw[v, u] 

            if z_mm == 0:
                self.get_logger().warn("Invalid depth at grey object pixel.")
                return

            x, y, z = self.pixel_to_camera_point(u, v, z_mm)
            point_cam = [x, y, z]
            point_world = self.transform_camera_to_world_tf(point_cam)

            if point_world is not None:
                # (180, 0, -90) [ 0.7071068, -0.7071068, 0.0, 0.0 ]
                # self.pick_target_pose = [point_world[0], point_world[1], point_world[2]-0.02, 0.7071068, -0.7071068, 0.0, 0.0] 
                self.pick_target_pose = [point_world[0], point_world[1], point_world[2]+0.0021, 1.0, 0.0, 0.0, 0.0]
                self.get_logger().info(f"Target pose calculated: {self.pick_target_pose}")

        if self.last_green_pixel is not None:
            u, v = self.last_green_pixel
            z_mm = depth_image_raw[v, u] 
            if z_mm == 0:
                self.get_logger().warn("Invalid depth at green object pixel.")
                return
            
            x, y, z = self.pixel_to_camera_point(u, v, z_mm)
            point_cam = [x, y, z]
            print("Green::::::",point_cam)
            point_world = self.transform_camera_to_world_tf(point_cam)
            if point_world is not None:
                # 0.05 is the offset of the green region
                self.place_target_pose = [point_world[0], point_world[1], point_world[2]+0.02, 1.0, 0.0, 0.0, 0.0]  
                self.get_logger().info(f"Target pose calculated: {self.place_target_pose}")

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

    def color_box_segmentation(self, rgb_image, color='green'):
        # Convert RGB image to HSV
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        if color == 'green':
            lower = np.array([40, 50, 50])
            upper = np.array([85, 255, 255])
            mask = cv2.inRange(hsv_image, lower, upper)
            mask_name = "green_mask_hsv.png"
        
        elif color == 'yellow':
            lower = np.array([15, 30, 60])
            upper = np.array([35, 150, 220])
            mask = cv2.inRange(hsv_image, lower, upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask_name = "yellow_mask_hsv.png"
        else:
            self.get_logger().warn(f"Unsupported color: {color}")
            return None
        
        # Save the mask for debugging
        cv2.imwrite(mask_name, mask)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        
        if color == 'yellow':
            # Filter contours by minimum area to avoid noise
            min_area = 100  # Adjust this threshold as needed
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            if len(valid_contours) == 0:
                return None
            
            # Sort contours by area (largest first)
            valid_contours.sort(key=cv2.contourArea, reverse=True)
            
            # If there are multiple contours, mask out the largest one and select the second largest
            # If there's only one contour, use it
            if len(valid_contours) > 1:
                # Create a copy of the mask
                modified_mask = mask.copy()
                
                # Fill the largest contour with black (mask it out)
                largest_contour = valid_contours[0]
                cv2.fillPoly(modified_mask, [largest_contour], 0)
                
                # Save the modified mask for debugging
                cv2.imwrite(f"modified_{mask_name}", modified_mask)
                
                # Use the second largest contour
                selected_contour = valid_contours[1]
            else:
                # Only one contour found, use it
                selected_contour = valid_contours[0]
        else:
            selected_contour = max(contours, key=cv2.contourArea)
            
        # Calculate centroid of the uppermost contour
        M = cv2.moments(selected_contour)
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
    
    def transform_camera_to_world_tf(self, point_cam, frame='camera_color_optical_frame'):
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
    
    def pre_coin(self, pick_up_target_pose, place_target_pose):
        self.publish_pose(pick_up_target_pose)
        self.publish_gripper_position(1.0)
        self.publish_pose(place_target_pose)
        self.publish_gripper_position(0.0)
        self.publish_pose(self.origin_pose)
        

def main(args=None):
    rclpy.init(args=args)
    node = PickPlace()
    # (180, 0, -90) [ 0.7071068, -0.7071068, 0.0, 0.0 ]
    # (180, -5, -90)[ 0.7064338, -0.7064338, 0.0308436, -0.0308435 ]
    # p0 = [0.15, 0.0, 0.25, 0.7071068, -0.7071068, 0.0, 0.0]
    # p1 = [0.15, 0.0, 0.25, 0.7064338, -0.7064338, 0.0308435, -0.0308436]
    p0 = [0.15, 0.0, 0.32, 1.0, 0.0, 0.0, 0.0]
    node.publish_pose(p0)
    # node.publish_pose(p1)
    time.sleep(3.0)

    try:
        node.get_logger().info("I tried!!!!!")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.get_logger().info("All actions done. Shutting down.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()