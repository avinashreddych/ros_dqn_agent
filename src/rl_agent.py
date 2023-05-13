#!/home/v-labsai-avinash-reddy/env/bin/python3

from rl_model import DQNAgent
from environment import Env
import torch
from sensor_msgs.msg import LaserScan
import torch.nn as nn
from std_msgs.msg import Float32MultiArray
from collections import deque
import rospy
import os
import json
import numpy as np
import random
import time
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

EPISODES = 3000

if __name__ == "__main__":
    try:
        rospy.init_node("rl_agent")

        pub_result = rospy.Publisher("result", Float32MultiArray, queue_size=5)
        pub_get_action = rospy.Publisher("get_action", Float32MultiArray, queue_size=5)
        result = Float32MultiArray()
        get_action = Float32MultiArray()

        state_size = 28
        action_size = 5
        rl_agent = DQNAgent(state_size=state_size, action_size=action_size)

        env = Env(action_size)

        scores, episodes = [], []
        global_step = 0

        start_time = time.time()
        for e in range(rl_agent.load_episode + 1, EPISODES):
            done = False
            state = env.reset()
            rl_agent.replay_buffer.clear()
            score = 0
            for t in range(rl_agent.episode_step):
                action = rl_agent.select_action(state)

                next_state, reward, done = env.step(action)

                rl_agent.replay_buffer.push(state, action, next_state, reward, done)

                if len(rl_agent.replay_buffer) >= rl_agent.train_start:
                    if global_step <= rl_agent.target_update:
                        rospy.loginfo("UPDATE POLICY NETWORK")
                        rl_agent.train(epochs=4, target_update=False)
                    else:
                        rospy.loginfo("UPDATE POLICY NETWORK")
                        rospy.loginfo("UPDATE TARGET NETWORK")
                        rl_agent.train(epochs=4, target_update=True)

                if global_step % rl_agent.target_update == 0:
                    rospy.loginfo("UPDATE TARGET NETWORK")
                    rl_agent.update_target_net()

                score += reward
                state = next_state
                get_action.data = [action, score, reward]
                pub_get_action.publish(get_action)

                if t > 500:
                    rospy.loginfo("Time out.")
                    done = True

                if (e % 10) == 0:
                    rl_agent.save_policy(extension=e)

                if done:
                    result.data = [score, np.max(rl_agent.q_value)]
                    pub_result.publish(result)
                    rl_agent.update_target_net()
                    scores.append(score)
                    episodes.append(e)
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    rospy.loginfo(
                        "Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d",
                        e,
                        score,
                        len(rl_agent.replay_buffer),
                        rl_agent.epsilon,
                        h,
                        m,
                        s,
                    )
                    param_keys = ["epsilon"]
                    param_values = [rl_agent.epsilon]
                    param_dictionary = dict(zip(param_keys, param_values))
                    break

                global_step += 1

            if rl_agent.epsilon > rl_agent.epsilon_min:
                rl_agent.epsilon *= rl_agent.epsilon_decay
    except rospy.ROSInterruptException:
        pass
