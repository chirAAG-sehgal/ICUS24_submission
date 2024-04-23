import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from tf.transformations import euler_from_quaternion
import math

# Define the boundary dimensions
BOUNDARY_X_MIN = 0
BOUNDARY_X_MAX = 20
BOUNDARY_Y_MIN = 0
BOUNDARY_Y_MAX = 27

# Define the drone's dimensions
DRONE_WIDTH = 1.05
DRONE_LENGTH = 1.05
DRONE_HEIGHT = 0.5  # Adjusted drone height to 0.5 units

# Define the exclusion regions
EXCLUSION_REGIONS = [
    {'x_min': 2.9, 'x_max': 5.1, 'y_min': 2.9, 'y_max': 24.1, 'z_min': 0, 'z_max': 100, 'reason': 'Exclusion Region 1'},
    {'x_min': 8.9, 'x_max': 11.1, 'y_min': 2.9, 'y_max': 24.1, 'z_min': 0, 'z_max': 100, 'reason': 'Exclusion Region 2'},
    {'x_min': 14.9, 'x_max': 17.1, 'y_min': 2.9, 'y_max': 24.1, 'z_min': 0, 'z_max': 100, 'reason': 'Exclusion Region 3'}
]

def in_exclusion_region(x, y, z):
    for region in EXCLUSION_REGIONS:
        if region['x_min'] <= x <= region['x_max'] and \
           region['y_min'] <= y <= region['y_max'] and \
           region['z_min'] <= z <= region['z_max']:
            return region['reason']
    return None

def model_states_callback(msg):
    drone_index = msg.name.index("red")  # Adjust model name
    drone_pose = msg.pose[drone_index]

    # Extract roll, pitch, and yaw angles from the drone's orientation
    orientation_q = drone_pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    roll, pitch, _ = euler_from_quaternion(orientation_list)

    # Calculate the coordinates of the four corners of the drone's base
    half_width = DRONE_WIDTH / 2
    half_length = DRONE_LENGTH / 2
    base_vertices = [
        (drone_pose.position.x + half_length * math.cos(roll) - half_width * math.sin(roll),
         drone_pose.position.y + half_length * math.sin(roll) + half_width * math.cos(roll),
         drone_pose.position.z),
        (drone_pose.position.x - half_length * math.cos(roll) - half_width * math.sin(roll),
         drone_pose.position.y - half_length * math.sin(roll) + half_width * math.cos(roll),
         drone_pose.position.z),
        (drone_pose.position.x - half_length * math.cos(roll) + half_width * math.sin(roll),
         drone_pose.position.y - half_length * math.sin(roll) - half_width * math.cos(roll),
         drone_pose.position.z),
        (drone_pose.position.x + half_length * math.cos(roll) + half_width * math.sin(roll),
         drone_pose.position.y + half_length * math.sin(roll) - half_width * math.cos(roll),
         drone_pose.position.z)
    ]

    # Check for collision with boundary
    for vertex in base_vertices:
        x, y, z = vertex
        if not (BOUNDARY_X_MIN <= x <= BOUNDARY_X_MAX and BOUNDARY_Y_MIN <= y <= BOUNDARY_Y_MAX):
            rospy.loginfo("\033[1;31mCollision detected due to boundary\033[0m")  # Collision detected due to boundary
            return

    # Check for collision with exclusion regions
    for vertex in base_vertices:
        x, y, z = vertex
        reason = in_exclusion_region(x, y, z)
        if reason:
            rospy.loginfo("\033[1;31mCollision detected due to: {}\033[0m".format(reason))  # Collision detected in exclusion region
            return

    # Check for collision with ground
    if drone_pose.position.z - DRONE_HEIGHT < 0.15:  # Considering the height of the drone
        rospy.loginfo("\033[1;31mCollision detected due to ground\033[0m")  # Collision detected with ground

def collision_detector():
    rospy.init_node('collision_detector', anonymous=True)
    rospy.Subscriber("/gazebo/model_states", ModelStates, model_states_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        collision_detector()
    except rospy.ROSInterruptException:
        pass
