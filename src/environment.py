#!/usr/local/bin/python3

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion
from respawn_goal import Respawn


class Env:
    def __init__(self, action_size) -> None:
        """
        setting the initial goal box
        subsribing to odometry topic
        publishing to cmd_vel topic
        initialised to reset the simulation- sets the robot to origin
        """
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.heading = 0.0
        self.action_size = action_size
        self.init_goal = True
        self.get_goal_box = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher("cmd_vel", Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber("odom", Odometry, self.get_odometry)
        self.reset_proxy = rospy.ServiceProxy("gazebo/reset_simulation", Empty)
        self.unpause_proxy = rospy.ServiceProxy("gazebo/unpause_physics", Empty)
        self.pause_proxy = rospy.ServiceProxy("gazebo/pause_physics", Empty)
        self.respawn_goal = Respawn()

    def get_odometry(self, odom):
        """
        this is a callback function for the subscriber topic -odom
        you recieve the odom data continously from the robot.

        this function observes the date and computes the necessary parameters
        of the environment
        """
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(
            self.goal_y - self.position.y, self.goal_x - self.position.x
        )

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def get_state(self, scan):
        scan_range = []
        heading = self.heading
        min_range = 0.13
        done = False
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float("Inf"):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)
        if min_range > min(scan_range) > 0:
            done = True

        current_distance = round(
            math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2
        )
        if current_distance < 0.2:
            self.get_goal_box = True

        # scan_range_state = scan_range[:: len(scan_range) // self.state_size] # this is for 360 degree scan
        # accept the state size when initialising the class if the robot scan is 360 degree
        return (
            scan_range
            + [heading, current_distance, obstacle_min_range, obstacle_angle],
            done,
        )

    def get_goal_distance(self):
        goal_distance = round(
            math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2
        )

        return goal_distance

    def set_reward(self, state, done, action):
        yaw_reward = []
        current_distance = state[-3]
        heading = state[-4]

        for i in range(self.action_size):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(
                0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0]
            )
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.goal_distance)
        reward = (round(yaw_reward[action] * 5, 2)) * distance_rate

        if done:
            rospy.loginfo("Collision!!")
            reward = -150
            self.pub_cmd_vel.publish(Twist())

        if self.get_goal_box:
            rospy.loginfo("Goal!!")
            reward = 200
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.get_position(True, delete=True)
            self.goal_distance = self.get_goal_distance()
            self.get_goal_box = False

        return reward

    def step(self, action):
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1) / 2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message("scan", LaserScan, timeout=5)
            except:
                pass

        state, done = self.get_state(data)
        reward = self.set_reward(state, done, action)

        return np.asarray(state), reward, done

    def reset(self):
        rospy.wait_for_service("gazebo/reset_simulation")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message("scan", LaserScan, timeout=5)
            except:
                pass

        if self.init_goal:
            self.goal_x, self.goal_y = self.respawn_goal.get_position()
            self.init_goal = False

        self.goal_distance = self.get_goal_distance()
        state, done = self.get_state(data)

        return np.asarray(state)
