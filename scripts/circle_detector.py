
import numpy as np
import scipy
import scipy.ndimage
import cv2
import math3d as m3d
import sys
import math
import time
import rospy
from dvs_msgs.msg import EventArray, Event
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from std_srvs.srv import SetBool
from URX.srv import desiredTCP, moveUR
from ur_node_rtde import urx_ros
import thread


#R_matrix = np.array([ [-0.9962979, -0.0477220,  0.0715060], [-0.0855636,  0.4698699, -0.8785790],  [0.0083290, -0.8814448, -0.4722137] ])
R_matrix = np.eye(3)

class event_circle_detector:

    def __init__(self, sizeX, sizeY, min_R, max_R):

        self.ros_node = rospy.init_node('circle_detector_node', anonymous=True)
        # self.robot = urx_ros("192.168.50.110")

        self.sizeX_ = sizeX
        self.sizeY_ = sizeY
        self.min_R_ = min_R
        self.max_R_ = max_R
        self.hough_grid = self.initialize_grid(sizeX, sizeY, min_R, max_R)


        self.perform_VS = False
        self.open_loop_tracking = False

        self.x_range = [80, 260]
        self.y_range = [65, 195]
        # self.x_range = [0, 346]
        # self.y_range = [0, 260]

        self.Z = 0.07

        self.max_threshold = 3

        self.bridge = CvBridge()

        self.last_grid_translate_t = -1

        self.camera_matrix = []
        self.dist_coef = []
        
        self.get_ros_camera_params("/dvs/camera_info")
        self.distortion_lookup = self.setup_undistortion_table()

        self.image_subs = rospy.Subscriber("/dvs/image_raw", Image, self.image_callback, queue_size=1)
        self.event_subs = rospy.Subscriber('/dvs/events', EventArray, self.event_callback, queue_size=1)
        self.velocity_subs = rospy.Subscriber("/dvs/vel", Twist, self.velocity_callback, queue_size=1)
        self.image_pubs = rospy.Publisher('/circle_detection', Image, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/ur_cmd_vel', Twist, queue_size=1)
        self.VS_start_srv = rospy.Service('startVS', SetBool, self.start_VS)
        self.open_loop_srv = rospy.Service('openLoop', SetBool, self.openLoopMode)
        self.explore_srv = rospy.Service('explore_srv', SetBool, self.exploreMode)

        self.moveTCPtCall = rospy.ServiceProxy('move_TCP', moveUR)

        self.move_x_sum = 0
        self.move_y_sum = 0

        self.rate = rospy.Rate(30)

        self.Kp = 3
        self.cmd_vel = Twist()

        self.xc = -1
        self.yc = -1
        self.rc = -1

        #self.spin()

    def exploreMode(self, mess):
        adjust_pose = Pose()
        adjust_pose.orientation.x = 0.
        adjust_pose.orientation.y = 0.
        adjust_pose.orientation.z = 0.
        adjust_pose.orientation.w = 1.
        if mess.data == True:
            adjust_pose.position.x = 0.02
            self.moveTCPtCall("davis", adjust_pose)
            adjust_pose.position.x = -0.02
            self.moveTCPtCall("davis", adjust_pose)         
            

    
    def openLoopMode(self, mess):
        if mess.data == True:
            self.open_loop_tracking = True
        else:
            self.open_loop_tracking = False

    def start_VS(self, mess):
        if mess.data == True:
            self.perform_VS = True
        else:
            self.perform_VS = False

    def spin(self):
        while not rospy.is_shutdown():
            self.xc, self.yc, self.rc = self.get_circle_location()
            if self.perform_VS:
                if self.xc>=0:
                    vx = self.Kp * (self.xc-(self.sizeX_/2)) * self.Z / self.camera_matrix[0,0]
                    vy = self.Kp * (self.yc-(self.sizeY_/2)) * self.Z / self.camera_matrix[1,1]
                    self.cmd_vel.linear.x = vx
                    self.cmd_vel.linear.y = vy
                    self.cmd_vel_pub.publish(self.cmd_vel)
                    # self.robot.move_robot_callback(self.cmd_vel)
                    # self.robot.robot_c.speedL(self.robot.cmd_velocity_vector, self.robot.acc, 0)
            
            self.rate.sleep()

    def get_ros_camera_params(self, camera_info_topic):
        camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
        self.camera_matrix = np.array(camera_info.K).reshape(3,3)
        self.dist_coef = np.array([camera_info.D])

    def initialize_grid(self, sizeX, sizeY, min_R, max_R):
        grid = np.zeros([sizeX, sizeY, max_R-min_R+1])
        return grid

    # def add_event(self, event):
    #     #extract x y limit for computing hough
    #     start_time = time.clock()
    #     x_min = max(0, event.x - self.max_R_);
    #     x_max = min(346, event.x + self.max_R_);
    #     y_min = max(0, event.y - self.max_R_);
    #     y_max = min(260, event.y + self.max_R_);
        
    #     j_list = []
    #     k_list = []
    #     r_list = []
    #     for j in range(x_min,x_max):
    #         for k in range(y_min,y_max):
    #             #compute r
    #             r = int( math.sqrt( np.power(float(event.x) - float(j),2) + np.power(float(event.y) - float(k),2) ) )
    #             if (r>=self.min_R_) and (r<=self.max_R_):
    #                 j_list.append(j)
    #                 k_list.append(k)
    #                 r_list.append(r-self.min_R_)
    #                 #self.hough_grid[j, k, r-self.min_R_] = self.hough_grid[j, k, r-self.min_R_] + 1.;
    #     # self.hough_grid[j_list, k_list, r_list] = self.hough_grid[j_list, k_list, r_list] + 1.;
    #     print("execution_time: ", time.clock() - start_time)

    def add_event(self, event):
        #extract x y limit for computing hough
        # start_time = time.clock()
        x_min = max(0, event.x - self.max_R_);
        x_max = min(346, event.x + self.max_R_);
        y_min = max(0, event.y - self.max_R_);
        y_max = min(260, event.y + self.max_R_);
        
        j_list, k_list  = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        r_list = np.add(-self.min_R_, np.sqrt(np.add( np.power(np.add(-event.x, j_list), 2), np.power(np.add(-event.y, k_list), 2)))).astype(int)

        correct_idx = np.where(np.logical_and(r_list >= 0, r_list < (self.max_R_-self.min_R_)))
        self.hough_grid[j_list[correct_idx], k_list[correct_idx], r_list[correct_idx]] = self.hough_grid[j_list[correct_idx], k_list[correct_idx], r_list[correct_idx]] + 1.;
        # print("execution_time: ", time.clock() - start_time)


    def add_event_array(self, event_array):
        calib_event = Event()
        for event in event_array.events:
            # start_time = time.clock()
            calib_event.x = np.round(self.distortion_lookup[event.x, event.y, 1])
            calib_event.y = np.round(self.distortion_lookup[event.x, event.y, 0])
            
            #if (event.x>100) and (event.x<250) and (event.y>60) and (event.y<230):
            if self.perform_VS:
                distance_from_center = np.sqrt( np.power(event.x - self.xc, 2) +  np.power(event.y - self.yc, 2))
                if distance_from_center < (1.5 * self.rc):
                    self.add_event(event)
            else:
                self.add_event(event)

            # print("execution_time: ", time.clock() - start_time)

    def translate_grid(self, camera_velocity, camera_matrix, depth):
        if self.last_grid_translate_t == -1:
            self.last_grid_translate_t = rospy.Time.now().to_sec()
        else:
            dt = rospy.Time.now().to_sec() - self.last_grid_translate_t
            if dt > 0.01:
                self.last_grid_translate_t = rospy.Time.now().to_sec()
                # print(dt)

                fx = camera_matrix[0,0];
                fy = camera_matrix[1,1];

                v_inertial_vector = np.matmul(R_matrix, np.array([camera_velocity.linear.x, camera_velocity.linear.y, camera_velocity.linear.z]))
                move_x = - dt * fx * v_inertial_vector[0] / depth;
                move_y = - dt * fy * v_inertial_vector[1] / depth;

                # print("G")
                # print(dt)
                # print(v_inertial_vector[0])
                # print(v_inertial_vector[1])
                # print(move_x)
                # print(move_y)

                self.move_x_sum = self.move_x_sum + move_x
                self.move_y_sum = self.move_y_sum + move_y
                
                if (abs(self.move_x_sum)>=1) or (abs(self.move_y_sum)>=1):


                    step_x = int(self.move_x_sum)
                    step_y = int(self.move_y_sum)
                    self.move_x_sum = self.move_x_sum - step_x
                    self.move_y_sum = self.move_y_sum - step_y

                    std_x = 0.4 * step_x;
                    std_y = 0.4 * step_y;
                
                    # start_time = time.clock()

                    if self.open_loop_tracking == False:
                        self.hough_grid = scipy.ndimage.filters.gaussian_filter(self.hough_grid, [std_x, std_y, 1]);
                    # print("execution_time: ", time.clock() - start_time)
                    #translate grid

                    # start_time = time.clock()
                    new_grid = np.zeros([self.sizeX_, self.sizeY_, self.max_R_-self.min_R_+1])
                    if step_x >= 0:
                        if step_y >= 0:
                            new_grid[step_x:, step_y:, :] = self.hough_grid[:(self.sizeX_-step_x), :(self.sizeY_-step_y), :]
                        else:
                            step_y = abs(step_y)
                            new_grid[step_x:, :(self.sizeY_-step_y), :] = self.hough_grid[:(self.sizeX_-step_x), step_y:, :]
                    else:
                        step_x = abs(step_x)
                        if step_y >= 0:
                            new_grid[:(self.sizeX_-step_x), step_y:, :] = self.hough_grid[step_x:, :(self.sizeY_-step_y), :]
                        else:
                            step_y = abs(step_y)
                            new_grid[:(self.sizeX_-step_x), :(self.sizeY_-step_y), :] = self.hough_grid[step_x:, step_y:, :]
                    self.hough_grid = new_grid
                    #self.hough_grid = scipy.ndimage.shift(self.hough_grid, [step_x, step_y, 0]); 
                    # print("execution_time: ", time.clock() - start_time)

    def get_circle_location(self):
        focused_grid = self.hough_grid[self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1], :]

        max_value = np.amax(focused_grid)

        if max_value > self.max_threshold:
            x_list, y_list, z_list = np.unravel_index(np.argmax(focused_grid), np.shape(focused_grid))
            if np.size(x_list) > 1:
                return x_list[0]+self.x_range[0], y_list[0]+self.y_range[0], z_list[0] + self.min_R_
            else:
                return x_list+self.x_range[0], y_list+self.y_range[0], z_list+self.min_R_
        else:
            return -1, -1, -1


    def plot_circle(self, input_image, circle_x, circle_y, circle_rad):
        return cv2.circle(input_image, (circle_x, circle_y), circle_rad, (255, 0, 0), 1)

    def image_callback(self, image_msg): 

        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        undistorted_image = cv_image#cv2.undistort(cv_image, self.camera_matrix, self.dist_coef)
        if np.shape(cv_image) == (260, 346, 3):
            if self.xc>=0:
                annotated_image = self.plot_circle(undistorted_image, self.xc, self.yc, self.rc)
                #cv2.imwrite('img.png', annotated_image)
                image_message = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")

                self.image_pubs.publish(image_message)
            else:
                image_message = self.bridge.cv2_to_imgmsg(undistorted_image, encoding="bgr8")
                self.image_pubs.publish(image_message)

        
    
    def event_callback(self, event_msg):
        # if event_msg.header.stamp.to_sec() < 1617168292.969481:
        if self.open_loop_tracking == False:
            self.add_event_array(event_msg)
        # self.hough_grid = np.multiply(0.9, self.hough_grid)

    def velocity_callback(self, vel_msg):
        self.translate_grid(vel_msg, self.camera_matrix, self.Z)

    def adjust_depth(self, target_depth):
        self.Z = target_depth

    def setup_undistortion_table(self):
        #adjusted_points = np.zeros([self.sizeX_, self.sizeY_, 2])
        
        map1x, map1y = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coef, np.eye(3), self.camera_matrix, (self.sizeY_, self.sizeX_), cv2.CV_16SC2)
        # print(map1x[259, 345, :])
        # print(map1y[0, 0])
        # rospy.sleep(100)
        # for i in range(self.sizeX_):
        #     for j in range(self.sizeY_):
        #         if (i == 0) and (j==0):
        #             print(np.array([[i, j]], np.float32))
        #             my_points = np.array([[i, j]], np.float32)
        #             adjusted_point = cv2.undistortPoints(np.array([[i, j]], np.float32), self.camera_matrix, self.dist_coef)
        #             print(adjusted_point)
        #             print(my_points)
        #             rospy.sleep(10)
        #         adjusted_point = cv2.undistortPoints(np.array([[i, j]], np.float32), self.camera_matrix, self.dist_coef)
        #         adjusted_points[i, j, :] = adjusted_point[0,0,:]

        # print(adjusted_points[110:120,100,0])
        adjusted_points = map1x
        
        return adjusted_points



if __name__ == '__main__':
    detector = event_circle_detector(346, 260, 12, 22)
    detector.spin()
    # thread.start_new_thread( detector.robot.run_node, () )
    # thread.start_new_thread( detector.spin, () )
    exit()



