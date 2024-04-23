#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Point, Transform, Twist,PointStamped
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
import os
import cv2
import numpy as np
import math
import threading  
from astar import wayp
from yolo8 import comparing
from ultralytics import YOLO
import time
from yolo8.forward import forward_function
from std_msgs.msg import String, Int32, Bool

print(os.getcwd())
os.chdir('/root/sim_ws/src/icuas24_competition/scripts')

current_position = Point()
current_orientation = {}
wayps = []
cv_image = None
cv_depth = None
depth_normalized = None
image_callback_msg=None
depth_callback_msg=None
output = []
beds=[]
tensor_dict= {}
rev_dict = {}
plant_name= ''
challenge_started_bool = False
plant_beds_bool = False
start_time=None
beds_lst = []
last_posn='NN'
fruit_class=0
dl_model_compl=[]
current_yaw=0
current_roll=0
total_dist=0
def publish_fruit_count(count):
    pub = rospy.Publisher('/fruit_count', Int32, queue_size=10, latch=True)
    rate = rospy.Rate(1)  
    rospy.loginfo(f"Publishing fruit count: {count}")
    pub.publish(count)
    
def dl_model_thread_forward(bed_id,model):
    global tensor_dict
    global rev_dict
    # print("*************************************")
    # print("Running foward model on bed : ",bed_id)
    # print("*************************************")
    
    forward_tensor,forward_pb_rev = forward_function(bed_id,model,fruit_class)
    
    
    tensor_dict[bed_id]=forward_tensor
    rev_dict[bed_id]=forward_pb_rev

def dl_model_thread(bed_id,model):
    global dl_model_compl
    global output
    # print("*************************************")
  #  print("Running backward on bed : ",bed_id)
    # print("*************************************")
    
    while bed_id not in tensor_dict:
        time.sleep(1)
    if rev_dict[bed_id] is not None and tensor_dict[bed_id] is not None:
             
        output.append([[bed_id]+comparing.dl_model(bed_id,model,tensor_dict[bed_id],fruit_class,rev_dict[bed_id])])
       # print(output)
    else:
        output.append([[bed_id],0,'No plant'])
            
    dl_model_compl.append(bed_id)


def gen_trajectory_point(point):
    trajectory_point = MultiDOFJointTrajectoryPoint()
    global wayps

    transform = Transform()
    transform.translation.x = point[0]
    transform.translation.y = point[1]
    transform.translation.z = point[2]
    transform.rotation.x = 0.0
    transform.rotation.y = 0.0
    transform.rotation.z = point[3]
    transform.rotation.w = 0.0
    
    twist = Twist()
    twist.linear.x=0
    twist.linear.y=0
    twist.linear.z=0

    trajectory_point.transforms.append(transform)
    trajectory_point.velocities.append(twist)
    trajectory_point.accelerations.append(twist)
    
    wayps=[transform.translation, point[4]]
    
    return trajectory_point

def click_image_thread(target_position,model,last_posn):
    global beds_lst
   # print("clicking image")
    if  target_position[1][1] != 'T' and target_position[1][1] != 'R' and target_position[1][1] != 'L' and int(target_position[1][:-1]) in beds:
            print(target_position[1],': ',time.time()-start_time,":::::::",total_dist)
            if (last_posn[1]=='R' or last_posn[1] == 'L' ) and  int(target_position[1][:-1]) in beds:
               # print("Coming from corner")
                while current_roll > 4 or current_roll < -4 :
                  #  print(current_roll)
                    time.sleep(0.05)
                if target_position[1][-1]=='F':
                    while current_yaw > 5 or current_yaw < -5:
                       # print(current_yaw)
                        time.sleep(0.05)
                if target_position[1][-1]=='B':
                    while current_yaw < 175 and current_yaw > -175:
                      #  print(current_yaw)
                        time.sleep(0.05)
                    
                
            elif last_posn[-1] == 'B' and target_position[1][-1] == 'F':
               # print("Turning around")
              #  print(target_position[1][-1])
                if target_position[1][-1]=='F':
                    print('str without loop',current_yaw)
                    while current_yaw > 5 or current_yaw < -5:
                       # print(current_yaw)
                        time.sleep(0.05)
                if target_position[1][-1]=='B':
                    while current_yaw < 175 and current_yaw > -175:
                     #   print(current_yaw)
                        time.sleep(0.05)
                    
               # print("Turning around")
                #time.sleep(3.1)
            else:
               # print("Other case")
                while current_roll > 4 or current_roll < -4 :
                   # print(current_roll)
                    time.sleep(0.05)
                
    # elif (last_posn[1]=='R' or last_posn[1] == 'L' ) and  int(target_position[1][:-1]) in beds:
    #     print("Coming from a turn")
    #     time.sleep(1.0)
    # else:
    #     time.sleep(1.15)  
        
    for i in range(25):
        image_depth_callback(image_callback_msg, depth_callback_msg,target_position,i)
        frame_rate = 60
        time.sleep(1/frame_rate)
    # Run DL model in a separate thread  
   # print(beds_lst)
    if int(target_position[1][:-1]) not in beds_lst:
     #  print(f"Adding {int(target_position[1][:-1])} to lst")
        beds_lst.append(int(target_position[1][:-1]))
        dl_thread_forward = threading.Thread(target=dl_model_thread_forward, args=(int(target_position[1][:-1]),model))
        dl_thread_forward.start()
    else:
        if target_position[1][-1] == 'B':
            dl_thread = threading.Thread(target=dl_model_thread, args=(int(target_position[1][:-1]),model))
            dl_thread.start()

def image_depth_callback(img_msg, depth_msg,target_position,i):
    try:
        

        # Convert ROS Image messages to OpenCV images
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
        cv_depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        
        cv_depth = np.nan_to_num(cv_depth)

        # Normalize the depth values to fit within the 8-bit range
        depth_normalized = cv2.normalize(cv_depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)
        
        # directory
        #  print("targetposition: ", target_position[1])
        directory = "images/" + str(target_position[1])
        if not os.path.exists(directory):
            os.makedirs(directory)

        # saving image
        
        image_location_name = "img" + "_" + str(target_position[1]) + "_" + str(i) + ".jpg"
        cv2.imwrite(f"{directory}/{image_location_name}", cv_image)

        # Save depth map
        
        depth_location_name = "depth" + "_" + str(target_position[1]) + "_" + str(i) + ".jpg"
        cv2.imwrite(f"{directory}/{depth_location_name}", depth_normalized)
        #rospy.loginfo("Image and Depth saved successfully!")
        
        # Save coordinate.txt file
        coordinate_file_path = os.path.join(directory, "coordinate"+ "_" + str(target_position[1]) + "_" + str(i) + ".txt")
        save_coordinate_file(coordinate_file_path, current_orientation)

    except Exception as e:
        rospy.logerr("Error processing the image and depth: %s", str(e))

def image_callback_threaded(msg):
    global image_callback_msg
    image_callback_msg = msg
 #   threading.Thread(target=image_depth_callback, args=(msg, depth_callback_msg)).start()

def depth_callback_threaded(msg):
    global depth_callback_msg
    depth_callback_msg = msg
 #   threading.Thread(target=image_depth_callback, args=(image_callback_msg, msg)).start()

def quaternion_to_euler(quaternion):
    """
    Convert quaternion to Euler angles in degrees (Roll, Pitch, Yaw).
    """
    
    x, y, z, w = quaternion

    # Convert quaternion to Euler angles (degrees)
    roll_x = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch_y = math.asin(2 * (w * y - z * x))
    yaw_z = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return math.degrees(roll_x), math.degrees(pitch_y), math.degrees(yaw_z)

def orientation_callback(data):
    global current_orientation
    global current_yaw
    global current_roll
    quaternion = (
        data.pose.pose.orientation.x,
        data.pose.pose.orientation.y,
        data.pose.pose.orientation.z,
        data.pose.pose.orientation.w
    )

    # Convert quaternion to Euler angles (degrees)
    roll, pitch, yaw = quaternion_to_euler(quaternion)
    current_yaw = yaw
    current_roll = roll
    current_orientation = {
        'roll': roll
    }
def calculate_distance2(point1, point2):
    zzz = math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)
   # rospy.loginfo(f"Distance {zzz}")
    return zzz
def calculate_distance(point1, point2):
    zzz = math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)
   # rospy.loginfo(f"Distance {zzz}")
    return zzz

def position_callback(data):
    global current_position
    current_position = data.point

def save_coordinate_file(file_path, orientation):
    with open(file_path, 'w') as file:
        file.write(f"{orientation['roll']}\n")

def publish_traj_pose(point):
    pub = rospy.Publisher('/red/position_hold/trajectory', MultiDOFJointTrajectoryPoint, queue_size=100, latch=True)
    
    rate = rospy.Rate(1)
    
    pub.publish(point)
    
def node_confirm(l,model):
    
    global last_posn
    global beds_lst
    global wayps
    rate = rospy.Rate(60)
    target_position = wayps
    #print(last_posn,'-->',target_position[1])
    if  target_position[1][1] != 'T' and target_position[1][1] != 'R' and target_position[1][1] != 'L' and int(target_position[1][:-1]) in beds:
        
        if last_posn == (target_position[1]):
           # print("Skipping this coordinate")
            last_posn=(target_position[1])
            return None
    
    i = 0
    if  target_position[1][1] != 'T' and target_position[1][1] != 'R' and target_position[1][1] != 'L' and int(target_position[1][:-1]) in beds:
       # print("running this loop")
        while calculate_distance(current_position, target_position[0]) > 1:
           # print(calculate_distance(current_position, target_position[0]))
            time_st = time.time()
            rate.sleep()
        while calculate_distance(current_position, target_position[0]) > 0.4:
            try:
                time_nd = time.time()
                
                if time_nd - time_st >1:
                   # print("breaking")
                    break
            except:
                pass
            rate.sleep()                                    
    else:
       # print("running this loop")
        while calculate_distance(current_position, target_position[0]) > 2:
            #print(calculate_distance(current_position, target_position[0]))
            time_st = time.time()
            rate.sleep()
        while calculate_distance(current_position, target_position[0]) > 1:
            time_nd = time.time()
            
            try:
                time_nd = time.time()
                
                if time_nd - time_st >1:
                 #   print("breaking")
                    break
            except:
                pass
            rate.sleep()
        
    
  #  print(target_position[1][1])
  #  print("want to run model:", (target_position[1][1] != 'T' and target_position[1][1] != 'R' and target_position[1][1] != 'L' and int(target_position[1][:-1]) in beds))
    if target_position[1][1] != 'T' and target_position[1][1] != 'R' and target_position[1][1] != 'L' and int(target_position[1][:-1]) in beds:
      #  print("Running model")
        thread_click_pic = threading.Thread(target=click_image_thread,args=(target_position,model,last_posn))
        thread_click_pic.start()
        
        last_posn=(target_position[1])
       
    last_posn=(target_position[1])
end = False
def cal_dist_thread():
    global total_dist
    while end==False:
        new_pos  = current_position
        time.sleep(0.1)
        total_dist += calculate_distance2(current_position,new_pos)
    
def plant_beds_callback():
    
    thread_dist= threading.Thread(target=cal_dist_thread)
    thread_dist.start()
    global end
    global beds 
    beds = [int(n) for n in beds]
  #  beds = [1,3,5,7,8,13,18,21,25]
   # beds = [14,17]
    beds = [14]
    time_astar = time.time()
    l = wayp(beds)
    print(l)
    time_end_astar = time.time()
    print("Time taken for algo: ",time_end_astar-time_astar)
    global wayps

    path = 'yolo8/runs/detect/train3_orsub_close/weights/best.pt'
    model = YOLO(path)
    

    last_point = [0,0,0,0,'ST']
    
    for point in l:
        if last_point[3]!=point[3]:
            # print("turn")
            last_point[3]= int(not last_point[3])
            last_point[4]= '000Z'
            temp_point = []
            temp_point.extend(last_point)
            #   print(temp_point)
            point_traj = gen_trajectory_point(temp_point)
            publish_traj_pose(point_traj)
            
            node_confirm(point,model)
            # print("Moving on to the next point ;")
            last_point = point
            time.sleep(2.3)
            
        point_traj = gen_trajectory_point(point)
        publish_traj_pose(point_traj)
        
        node_confirm(point,model)
        # print("Moving on to the next point ;")
        last_point = point

        wayps = []
    
    print("-----------------------------------------------")
    print(dl_model_compl)
    print(beds)
    print("-----------------------------------------------")
    
    # while math.sqrt((current_position.x - l[-1][-1][0]) ** 2 + (current_position.y - l[-1][-1][1]) ** 2 + (current_position.z - l[-1][-1][2]) ** 2) > 0.7:
    #     time.sleep(0.1)   
    #print(dl_model_compl)
   # print(beds)
    while len(dl_model_compl)!=len(beds):
        time.sleep(0.2)
        
    end_time = time.time()
    
    answer=0
    
    for i in output:
        print(i)
        try:
            answer+=i[0][1]
        except:
            continue
    print(f"Total distance: {total_dist}")
    print(f"Total time : {end_time - start_time}")
    print("Fruit Count: ",answer)
    end = True
    publish_fruit_count(answer)

def challenge_started(data):
   # print(data.data)
    global challenge_started_bool
    if data.data ==True:    
        challenge_started_bool = True
      #  print("Challenge Started")

def plant_beds(data):
    global start_time
    global plant_beds_bool
    global plant_name
    global beds
    global fruit_class
    
    if challenge_started_bool==True:
        start_time=time.time()
      #  print("plant beds published")
        print("Initiating Code")
        beds_data = data.data
        beds_list = beds_data.split(" ")
        print(beds_list)
        plant_name = beds_list[0]
        if plant_name == 'Pepper':
            fruit_class = 1
        elif plant_name == 'Tomato':
            fruit_class = 0
        else: 
            fruit_class = 2
        #print(fruit_class)
            
        beds = beds_list[1:]
        plant_beds_callback()
        
if __name__ == '__main__':
    rospy.init_node('coordinate_publisher', anonymous=True)
    rospy.Subscriber('/red/challenge_started', Bool, challenge_started)
    rospy.Subscriber('/red/plants_beds', String, plant_beds)
    rospy.Subscriber('/red/camera/color/image_raw', Image, image_callback_threaded)
    rospy.Subscriber('/red/camera/depth/image_raw', Image, depth_callback_threaded)
    rospy.Subscriber('/red/position', PointStamped, position_callback)
    rospy.Subscriber('/red/odometry', Odometry, orientation_callback)
   # plant_beds_callback()
    rospy.spin()
    
###########
