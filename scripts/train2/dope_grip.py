import cv2
from src.cameras.camera import Camera
from src.object_detection.detect import find_object
from src.robots.urx_robot import UR_Robot as Robot
from src.utils.ur_parser import UR_Parser as Parser
from src.utils.persistent_data import create_logger
from src.constants.commons import DEMARCATOR
from src.train2.find_object import find_object, Detector
from src.train2.detect import DopeNode
from src.object_detection.localization import Point2d, Point3d, convert_depth_pixel_to_metric_coordinate, get_depth_at_pixel
import math
import numpy as np
from pyquaternion import Quaternion

pick = 1
connect = 1
debug = 0

def main():
    # Whether to log out orientation or centroid
    filename = "ori.txt"
    if debug:
        with open(filename, "a") as f:
            f.write(f'New data\n')
    num = 10
    for i in range(num): # Wait for it to stabilize
        ret, imgs, frames = cap.get_frame()
        color_frame, depth_frame = frames

        if not ret:
            break

        intrinsics = cap.get_camera_intrinsics()
        camera_info = {'camera_matrix' : {'data' : [intrinsics.fx, 0, intrinsics.ppx, 0,0, intrinsics.fy, intrinsics.ppy, 0,0,0,1 ,0]}}
        model_coordinates, rotation, color_img, duration, points = find_object(imgs[0], camera_info, detector.dope_node,opt) # Uses dope inference

        # If detected
        if model_coordinates is not None:
            # Convert model coordinates output from centimeters to millimetres 
            model_coordinates = [i * 0.01 for i in model_coordinates]
            # Get 3D centroid predicted by model as 2D point in image 
            x_center,y_center = points[-1]
            # Convert 2D point to real 3D Coordinate using realsense (it will output the 3D XYZ distance where the light hits the surface of the object)
            center3d = convert_depth_pixel_to_metric_coordinate(
                get_depth_at_pixel(depth_frame, x_center, y_center),
                x_center,
                y_center,
                intrinsics,
            )
            if debug:
                print("3D coordinates centroid from model:", model_coordinates)
                print("3D coordinates from realsense:", center3d)
            # Send 3D coordinate containing X,Y from model and minimum Z distance between model's depth and realsense's depth
            coord_3d = Point3d(model_coordinates[0], model_coordinates[1], min(model_coordinates[2], center3d[2]))
            q = Quaternion(x=rotation[0], y=rotation[1], z=rotation[2], w=rotation[3])
            yaw,pitch,roll= q.yaw_pitch_roll
            # As model turns 180 degrees, yaw and roll will change from pi/2 to -pi/2 due to euler angle repesentation
            if yaw > 0 and roll > 0:
                coord_3d.angle = math.pi/2 + pitch
            elif yaw < 0 and roll < 0:
                coord_3d.angle = -math.pi/2 + -pitch
            # Time taken for inference
            print(f'duration : {duration}')
            if debug:
                with open(filename, "a") as f:
                    if model_coordinates:
                        f.write(f'{coord_3d.x_c} {coord_3d.y_c} {coord_3d.z_c}')
                        f.write(f'{yaw} {pitch} {roll} \n')

                vid_writer.write(color_img)
        cv2.imshow("inference", color_img)
        cv2.waitKey(1)
    if model_coordinates is not None:
        if pick:
            dynamicScript = """"
def __main__():
    set_standard_analog_input_domain(0, 1)
    set_standard_analog_input_domain(1, 1)
    set_tool_analog_input_domain(0, 1)
    set_tool_analog_input_domain(1, 1)
    set_analog_outputdomain(0, 0)
    set_analog_outputdomain(1, 0)
    set_tool_voltage(0)
    set_input_actions_to_default()
    set_tcp(p[0.0,0.0,0.0,0.0,0.0,0.0])
    set_payload(0.0)
    set_gravity([0.0, 0.0, 9.82])
    # begin: URCap Installation Node
    #   Source: RG - On Robot, 1.9.0, OnRobot A/S
    #   Type: RG Configuration
    global measure_width=0
    global grip_detected=False
    global lost_grip=False
    global zsysx=0
    global zsysy=0
    global zsysz=0.06935
    global zsysm=0.7415
    global zmasx=0
    global zmasy=0
    global zmasz=0.18659
    global zmasm=0
    global zmasm=0
    global zslax=0
    global zslay=0
    global zslaz=0
    global zslam=0
    global zslam=0
    thread lost_grip_thread():
        while True:
        set_tool_voltage(24)
        if True ==get_digital_in(9):
            sleep(0.024)
            if True == grip_detected:
            if False == get_digital_in(8):
                grip_detected=False
                lost_grip=True
            end
            end
            set_tool_analog_input_domain(0, 1)
            set_tool_analog_input_domain(1, 1)
            zscale = (get_analog_in(2)-0.026)/2.9760034
            zangle = zscale*1.57079633+-0.08726646
            zwidth = 5.0+110*sin(zangle)
            global measure_width = (floor(zwidth*10))/10-9.2
        end
        sync()
        end
    end
    lg_thr = run lost_grip_thread()
    def RG2(target_width=110, target_force=40, payload=0.0, set_payload=False, depth_compensation=False, slave=False):
        grip_detected=False
        if slave:
        slave_grip_detected=False
        else:
        master_grip_detected=False
        end
        timeout = 0
        timeout_limit = 750000
        while get_digital_in(9) == False:
        if timeout > timeout_limit:
            break
        end
        timeout = timeout+1
        sync()
        end
        def bit(input):
        msb=65536
        local i=0
        local output=0
        while i<17:
            set_digital_out(8,True)
            if input>=msb:
            input=input-msb
            set_digital_out(9,False)
            else:
            set_digital_out(9,True)
            end
            if get_digital_in(8):
            out=1
            end
            sync()
            set_digital_out(8,False)
            sync()
            input=input*2
            output=output*2
            i=i+1
        end
        return output
        end
        target_width=target_width+9.2
        if target_force>40:
        target_force=40
        end
        if target_force<4:
        target_force=4
        end
        if target_width>110:
        target_width=110
        end
        if target_width<0:
        target_width=0
        end
        rg_data=floor(target_width)*4
        rg_data=rg_data+floor(target_force/2)*4*111
        rg_data=rg_data+32768
        if slave:
        rg_data=rg_data+16384
        end
        bit(rg_data)
        if depth_compensation:
        finger_length = 55.0/1000
        finger_heigth_disp = 5.0/1000
        center_displacement = 7.5/1000
        start_pose = get_forward_kin()
        set_analog_inputrange(2, 1)
        zscale = (get_analog_in(2)-0.026)/2.9760034
        zangle = zscale*1.57079633+-0.08726646
        zwidth = 5.0+110*sin(zangle)
        start_depth = cos(zangle)*finger_length
        sleep(0.016)
        timeout = 0
        while get_digital_in(9) == True:
            timeout=timeout+1
            sleep(0.008)
            if timeout > 20:
            break
            end
        end
        timeout = 0
        timeout_limit = 750000
        while get_digital_in(9) == False:
            zscale = (get_analog_in(2)-0.026)/2.9760034
            zangle = zscale*1.57079633+-0.08726646
            zwidth = 5.0+110*sin(zangle)
            measure_depth = cos(zangle)*finger_length
            compensation_depth = (measure_depth - start_depth)
            target_pose = pose_trans(start_pose,p[0,0,-compensation_depth,0,0,0])
            if timeout > timeout_limit:
            break
            end
            timeout=timeout+1
            servoj(get_inverse_kin(target_pose),0,0,0.008,0.01,2000)
            if point_dist(target_pose, get_forward_kin()) > 0.005:
            popup("Lower grasping force or max width",title="RG-lag threshold exceeded", warning=False, error=False, blocking=False)
            end
        end
        nspeed = norm(get_actual_tcp_speed())
        while nspeed > 0.001:
            servoj(get_inverse_kin(target_pose),0,0,0.008,0.01,2000)
            nspeed = norm(get_actual_tcp_speed())
        end
            stopj(2)
        end
        if depth_compensation==False:
        timeout = 0
        timeout_count=20*0.008/0.008
        while get_digital_in(9) == True:
            timeout = timeout+1
            sync()
            if timeout > timeout_count:
            break
            end
        end
        timeout = 0
        timeout_limit = 750000
        while get_digital_in(9) == False:
            timeout = timeout+1
            sync()
            if timeout > timeout_limit:
            break
            end
        end
        end
        sleep(0.024)
        if set_payload:
            if slave:
            if get_analog_in(3) < 2:
                zslam=0
            else:
                zslam=payload
            end
            else:
            if get_digital_in(8) == False:
                zmasm=0
            else:
                zmasm=payload
            end
            end
            zload=zmasm+zslam+zsysm
            set_payload(zload,[(zsysx*zsysm+zmasx*zmasm+zslax*zslam)/zload,(zsysy*zsysm+zmasy*zmasm+zslay*zslam)/zload,(zsysz*zsysm+zmasz*zmasm+zslaz*zslam)/zload])
        end
        master_grip_detected=False
        master_lost_grip=False
        slave_grip_detected=False
        slave_lost_grip=False
        if True == get_digital_in(8):
            master_grip_detected=True
        end
        if get_analog_in(3)>2:
            slave_grip_detected=True
        end
        grip_detected=False
        lost_grip=False
        if True == get_digital_in(8):
            grip_detected=True
        end
        zscale = (get_analog_in(2)-0.026)/2.9760034
        zangle = zscale*1.57079633+-0.08726646
        zwidth = 5.0+110*sin(zangle)
        global measure_width = (floor(zwidth*10))/10-9.2
        if slave:
            slave_measure_width=measure_width
        else:
            master_measure_width=measure_width
        end
        return grip_detected
        end
    set_tool_voltage(24)
    set_tcp(p[0,0,0.18659,0,-0,3.14])
    RG2(target_width=110, target_force=40, payload=0.5, set_payload=False, depth_compensation=False, slave=False)
    sync()
    sleep(0.5)
    # END OF INIT CODE
            """
            dynamicScript += Parser.get_dynamic_pick_script(coord_3d)
            is_sent = robot.handle_program_script_dynamic(dynamicScript)
            print(f'sent {coord_3d.x_c} {coord_3d.y_c} {coord_3d.z_c} with rotation of {coord_3d.angle}')
    if connect:
        robot.stop()
        

if __name__ == "__main__":
    # Changes the parameters below to run different models under different settings
    MODEL = "test"

    cap = Camera('realsense')
    
    # Initializing and connecting to robot
    HOST = "192.168.1.246"
    PORT = 30003
    # Loads parsed program scripts
    scripts = []
    if connect:
        logger = create_logger("models")
        robot = Robot(HOST, PORT, scripts, 1000, logger, "default")
        print("Connected to robot")

    FPS, VIS_IMAGE_SIZE = cap.get_params()
    save_path = f"models/{MODEL}/inference.mp4"
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, VIS_IMAGE_SIZE)

    # Set configurations for model
    opt = lambda x : None
    opt.config = "./src/train2/config_inference/config_pose.yaml"
    opt.camera = "./src/train2/config_inference/camera_info.yaml"
    opt.vidpath = "./vids"
    opt.outf = "./vids"
    opt.realsense = True
    opt.resize = True

    # Create Dope detector passing in the configurations
    detector = Detector(opt)

    try:
        main()
    finally:
        vid_writer.release()
        if connect:
            robot.stop()
        cap.stop()