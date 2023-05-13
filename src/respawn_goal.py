#!/usr/local/bin/python3
import rospy
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
import sys


class Respawn:
    def __init__(self) -> None:
        """
        the model used here is a basic square block
        """
        self.model_path = "/home/v-labsai-avinash-reddy/Desktop/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf"
        self.f = open(self.model_path, "r")
        self.model = self.f.read()
        self.f.close()
        self.goal_position = Pose()
        self.init_goal_x = 0.6
        self.init_goal_y = 0.0
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.model_name = "goal"
        self.obstacle_1 = [0.6, 0.6]
        self.obstacle_2 = [0.6, -0.6]
        self.obstacle_3 = [-0.6, 0.6]
        self.obstacle_4 = [-0.6, -0.6]
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber(
            "gazebo/model_states", ModelStates, self.check_model_
        )
        self.check_model = False
        self.index = 0

    def check_model_(self, model):
        """
        check_model is True if goal is decided.
        else False
        """
        self.check_model = False
        for model_name in model.name:
            if model_name == "goal":
                self.check_model = True

    def respwan_model(self):
        """
        if the goal is not present in the environment,
        then spawn a goal square box in the environment at the decided
        goal position
        """
        while True:
            if not self.check_model:
                rospy.wait_for_service("gazebo/spawn_sdf_model")
                spawn_model_prox = rospy.ServiceProxy(
                    "gazebo/spawn_sdf_model", SpawnModel
                )
                spawn_model_prox(
                    self.model_name,
                    self.model,
                    "robots_name_space",
                    self.goal_position,
                    "world",
                )
                rospy.loginfo(
                    f" Goal Position : {self.goal_position.position.x}, {self.goal_position.position.y}"
                )
                break
            else:
                pass

    def delete_model(self):
        """
        once the robot reaches the goal, new goal position is created
        so the old goal box should be deleted
        """
        while True:
            if self.check_model:
                rospy.wait_for_service("gazebo/delete_model")
                del_model_prox = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
                del_model_prox(self.model_name)
                break
            else:
                pass

    def get_position(self, position_check=False, delete=False):
        """
        randomly setting the goal position with constraints
        new_goal should not be inside obstacles
        new_goal should not be at origin
        new goal should not be nearer to the previous goal
        """
        if delete:
            self.delete_model()

        while position_check:
            goal_x = random.randrange(-16, 17) / 10.0
            goal_y = random.randrange(-16, 17) / 10.0
            if (
                abs(goal_x - self.obstacle_1[0]) <= 0.4
                and abs(goal_y - self.obstacle_1[1]) <= 0.4
            ):
                position_check = True
            elif (
                abs(goal_x - self.obstacle_2[0]) <= 0.4
                and abs(goal_y - self.obstacle_2[1]) <= 0.4
            ):
                position_check = True
            elif (
                abs(goal_x - self.obstacle_3[0]) <= 0.4
                and abs(goal_y - self.obstacle_3[1]) <= 0.4
            ):
                position_check = True
            elif (
                abs(goal_x - self.obstacle_4[0]) <= 0.4
                and abs(goal_y - self.obstacle_4[1]) <= 0.4
            ):
                position_check = True
            elif abs(goal_x - 0.0) <= 0.4 and abs(goal_y - 0.0) <= 0.4:
                position_check = True
            else:
                position_check = False

            if (
                abs(goal_x - self.last_goal_x) < 1
                and abs(goal_y - self.last_goal_y) < 1
            ):
                position_check = True

            self.goal_position.position.x = goal_x
            self.goal_position.position.y = goal_y

        time.sleep(0.5)
        self.respwan_model()
        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y
