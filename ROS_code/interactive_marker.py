#!/usr/bin/env python

import rospy
import tf.transformations
import numpy as np

from interactive_markers.interactive_marker_server import \
    InteractiveMarkerServer, InteractiveMarkerFeedback
from visualization_msgs.msg import InteractiveMarker, \
    InteractiveMarkerControl
from geometry_msgs.msg import PoseStamped, Quaternion
from franka_msgs.msg import FrankaState

def generate_circle_trajectory(radius, num_points):
    circle_trajectory = []  

    center_y = 0.1  # Center of the circle in x-coordinate
    center_z = 0.4 # Center of the circle in y-coordinate
    
    for i in range(num_points):
        # theta = 2 * np.pi * i / num_points
        theta = i / 80
        y = center_y + radius * np.cos(theta)
        z = center_z + radius * np.sin(theta)

        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "panda_link0"  # Adjust the frame_id as needed
        pose.pose.position.x = 0.45
        pose.pose.position.y = y
        pose.pose.position.z = z  # Adjust the z-coordinate as needed

        circle_trajectory.append(pose)

    return circle_trajectory


def generate_spiral_trajectory(radius, num_points):

    spiral_trajectory = []  
    center_y = 0.3  # Center of the circle in x-coordinate
    center_x = 0.3 # Center of the circle in y-coordinate
    center_z = 0.2
    z_speed = 0.2
    
    for i in range(num_points):
        # theta = 2 * np.pi * i / num_points
        theta = i / 40
        x = center_x + radius * np.sin(theta) 
        y = center_y + radius * np.cos(theta)
        z = center_z + z_speed * i / num_points

        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "panda_link0"  # Adjust the frame_id as needed
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z  # Adjust the z-coordinate as needed

        spiral_trajectory.append(pose)

    return spiral_trajectory


def read_trajectory_from_file(file_path):
    poses = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip():  # Check if line is not empty
                values = list(map(float, line.split()))  # Assuming each line contains x, y, z, qx, qy, qz, qw separated by spaces
                x, y, z, qx, qy, qz, qw = values[:7]  # Extracting position and quaternion values
                pose = PoseStamped()
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = "base_link"  # Adjust the frame_id as needed
                pose.pose.position.x = x
                pose.pose.position.y = y 
                pose.pose.position.z = z 
                pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)  # Assigning quaternion values
                poses.append(pose)
    return poses

if __name__ == "__main__":
    
    rospy.init_node("trajectory_generator")
    
    traj_publisher = rospy.Publisher('equilibrium_pose',PoseStamped,queue_size=20)
    radius = 0.08
    num_points = 1000

    # trajectory = generate_circle_trajectory(radius,num_points)
    
    trajectory = generate_spiral_trajectory(radius,num_points)

    file_path = "/home/manuel/Scrivania/TASK_VISUALIZATION/robot_16_data_MS1.txt"  # Adjust the file path accordingly

    trajectory = read_trajectory_from_file(file_path)

    rate=rospy.Rate(30)
    while not rospy.is_shutdown():
        for point in trajectory:
            traj_publisher.publish(point)
            rate.sleep()    


    rospy.spin()
