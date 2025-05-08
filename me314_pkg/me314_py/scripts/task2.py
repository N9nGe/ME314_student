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

        self.last_red_pixel = None
        self.last_blue_pixel = None
        
        self.fx = 605.763671875
        self.fy = 606.1971435546875
        self.cx = 324.188720703125
        self.cy = 248.70957946777344

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pick_target_pose = None
        self.place_target_pose = None
        self.object_locked = False
        self.origin_pose = [0.16664958131555577, 0.0028170095361637172, 0.18776892332246256, 1.0, 0.0, 0.0, 0.0]  # Default origin pose

        self.pre_timer = self.create_timer(1.0, self.pre_timer_callback)
        self.timer = self.create_timer(1.0, self.timer_callback)

        self.pre_task2_done = False
        self.task2_done = False
        

    def pre_timer_callback(self):
        if not self.pre_task2_done:
            if self.pick_target_pose and self.place_target_pose:
                diff = abs(self.place_target_pose[0] - self.current_arm_pose.position.x) \
                    + abs(self.place_target_pose[1] - self.current_arm_pose.position.y) \
                    + abs(self.place_target_pose[2] - self.current_arm_pose.position.z)
                if diff < 1e-2:
                        self.pre_task2_done = True
                if not self.object_locked:
                    self.get_logger().info("Timer triggered Pre plug.")
                    self.pre_execute_plug_and_hole(self.pick_target_pose, self.place_target_pose)
                    self.object_locked = True

    def timer_callback(self):
        if self.pre_task2_done and not self.task2_done:
            if self.pick_target_pose and self.place_target_pose:
                self.get_logger().info("Timer triggered plug.")
                start_point = [self.pick_target_pose[0], self.pick_target_pose[1]]
                end_point = [self.place_target_pose[0], self.place_target_pose[1]]
                unit_vector = self.unit_vector(start_point, end_point) 

                # tried to plug in
                current_xy = [self.current_arm_pose.position.x, self.current_arm_pose.position.y]
                new_pose = [self.current_arm_pose.position.x,
                            self.current_arm_pose.position.y,
                            self.current_arm_pose.position.z,
                            self.current_arm_pose.orientation.x,
                            self.current_arm_pose.orientation.y,
                            self.current_arm_pose.orientation.z,
                            self.current_arm_pose.orientation.w
                ]
                if self.current_arm_pose.position.z > 0.13:
                    self.execute_plug_and_hole(current_xy, unit_vector, new_pose)
                else:
                    self.task2_done = True
                    self.publish_gripper_position(0.0)
                    self.publish_pose(self.origin_pose)

    def arm_pose_callback(self, msg: Pose):
        self.current_arm_pose = msg

    def gripper_position_callback(self, msg: Float64):
        self.current_gripper_position = msg.data

    def arm_executing_callback(self, msg: Bool):
        self.arm_executing = msg.data

    def collision_callback(self, msg: Bool):
        self.collision_check = msg.data
    
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

        # get the center of the red box
        red_center = self.color_box_segmentation(raw_image, 'red')
        if red_center is not None:
            cx, cy = red_center
            self.get_logger().info(f"Red object center: ({cx}, {cy})")

            # Test code
            rgb_image_marked = raw_image.copy()
            cv2.circle(rgb_image_marked, (cx, cy), 8, (0, 255, 0), thickness=2)  # 绿色圆圈
            cv2.imwrite('rgb_image_marked.png', rgb_image_marked)

            # store the current coordinate
            self.last_red_pixel = (cx, cy)
        else:
            self.get_logger().info("Red object not found.")

        blue_center = self.color_box_segmentation(raw_image, 'blue')
        if blue_center is not None:
            cx, cy = blue_center
            self.get_logger().info(f"blue object center: ({cx}, {cy})")

            # Test code
            blue_image_marked = raw_image.copy()
            cv2.circle(blue_image_marked, (cx, cy), 8, (255, 0, 0), thickness=2)  # 绿色圆圈
            cv2.imwrite('blue_image_marked.png', blue_image_marked)

            self.last_blue_pixel = (cx, cy)
        else:
            self.get_logger().info("blue object not found.")

    def depth_image_callback(self, msg: Image):
        if self.object_locked or self.last_red_pixel is None or self.arm_executing:
            return
        depth_image_raw = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        # for display 
        depth_image = cv2.normalize(depth_image_raw, None, 0, 255, cv2.NORM_MINMAX)
        depth_image = depth_image.astype(np.uint8)
        # cv2.imwrite('depth_display.png', depth_   image)
        if self.last_blue_pixel is not None:
            u, v = self.last_red_pixel
            z_mm = depth_image_raw[v, u] 

            if z_mm == 0:
                self.get_logger().warn("Invalid depth at red object pixel.")
                return

            x, y, z = self.pixel_to_camera_point(u, v, z_mm)
            point_cam = [x, y, z]
            point_world = self.transform_camera_to_world_tf(point_cam)

            if point_world is not None:
                self.pick_target_pose = [point_world[0], point_world[1], point_world[2]-0.01, 1.0, 0.0, 0.0, 0.0]  
                self.get_logger().info(f"Target pose calculated: {self.pick_target_pose}")
        
        if self.last_blue_pixel is not None:
            u, v = self.last_blue_pixel
            z_mm = depth_image_raw[v, u] 
            if z_mm == 0:
                self.get_logger().warn("Invalid depth at blue object pixel.")
                return
            
            x, y, z = self.pixel_to_camera_point(u, v, z_mm)
            point_cam = [x, y, z]
            point_world = self.transform_camera_to_world_tf(point_cam)
            if point_world is not None:
                # 0.05 is the offset of the blue region
                self.place_target_pose = [point_world[0], point_world[1], point_world[2]+0.115, 1.0, 0.0, 0.0, 0.0]  
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

    def color_box_segmentation(self, rgb_image, color='red'):
        # Convert RGB image to HSV
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        if color == 'red':
            lower1 = np.array([0, 100, 100])
            upper1 = np.array([10, 255, 255])

            lower2 = np.array([160, 100, 100])
            upper2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv_image, lower1, upper1)
            mask2 = cv2.inRange(hsv_image, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)

            mask_name = "red_mask_hsv.png"

        elif color == 'blue':
            lower = np.array([100, 100, 100])
            upper = np.array([130, 255, 255])
            mask = cv2.inRange(hsv_image, lower, upper)

            mask_name = "blue_mask_hsv.png"
        
        else:
            self.get_logger().warn(f"Unsupported color: {color}")
            return None

        # Save the mask for debugging
        cv2.imwrite(mask_name, mask)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
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

    def unit_vector(self, pointA, pointB):
        # Convert to NumPy arrays
        A = np.array(pointA, dtype=float)
        B = np.array(pointB, dtype=float)

        # Compute the direction vector
        vec = B - A

        # Compute the magnitude (Euclidean norm)
        norm = np.linalg.norm(vec)

        if norm == 0:
            raise ValueError("Points A and B are the same. Cannot compute unit vector.")

        # Normalize the vector
        return vec / norm

    def pre_execute_plug_and_hole(self, pick_up_target_pose, place_target_pose):
        # pick up the red cylinder and put it above the hole
        self.publish_pose(pick_up_target_pose)
        self.publish_gripper_position(1.0)
        pick_up_target_pose[2] += 0.05
        self.publish_pose(pick_up_target_pose)
        self.publish_pose(place_target_pose)
        
    def execute_plug_and_hole(self, current_xy, unit_vector, new_pose):
        if not self.arm_executing:
            if self.collision_check:
                current_xy = list(np.array(current_xy) + unit_vector * 0.1)
                new_pose[0] = current_xy[0]
                new_pose[1] = current_xy[1]
                self.publish_pose(new_pose)
                time.sleep(2.0)
            
            new_pose[2] = self.current_arm_pose.position.z - 0.01
            self.publish_pose(new_pose)
            time.sleep(2.0)
        

def main(args=None):
    rclpy.init(args=args)
    node = PickPlace()
    p0 = [0.15, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]

    node.publish_pose(p0)
    time.sleep(5.0)

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