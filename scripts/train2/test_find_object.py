import cv2
from inference import DopeNode
from find_object import Detector, find_object
import math

pick = 0


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def main():
    while True: # Wait for it to stabilize
        frame = detector.get_frames()
        detector.get_rs_intrinsics()
        coordinates, orientation, color_img, duration = find_object(frame, detector.camera_info, detector.dope_node,opt) # Uses dope inference
        if orientation is not None:
            orientation = euler_from_quaternion(orientation[0], orientation[1],orientation[2],orientation[3])
        print("coordinates:", orientation)
        print(f'duration : {duration}')

        #vid_writer.write(color_img)
        cv2.imshow("inference", color_img)
        cv2.waitKey(1)

if __name__ == "__main__":
    #vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, VIS_IMAGE_SIZE)

    print("hey")
    opt = lambda x : None
    opt.config = "config_inference/config_pose.yaml"
    opt.camera = "config_inference/camera_info.yaml"
    opt.vidpath = "./vids"
    opt.outf = "./vids"
    opt.realsense = True
    opt.resize = True

    detector = Detector(opt)

    try:
        main()
    finally:
        pass
        #vid_writer.release()
        #cap.stop()
