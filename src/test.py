#!/home/v-labsai-avinash-reddy/env/bin/python3

import rospy
from sensor_msgs.msg import LaserScan


rospy.init_node("scan_values")

try:
    data = rospy.wait_for_message("scan", LaserScan, timeout=5)
except:
    print("error")

print(len(data.ranges))
print(data.ranges)
