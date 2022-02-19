import argparse
import yaml 
import glob 
import os 
import shutil
import time        

import cv2
import numpy as np
import pyrealsense2 as rs
from inference import DopeNode

class Detector():
    def __init__(self,opt):
        self.opt = opt
        self.create_output_folder()
        # Load initial configs
        self.pose_config = self.load_configs()
        if opt.realsense:
            self.pipeline, self.config, self.profile = self.setup_realsense()
        else:
            self.cap = cv2.VideoCapture(opt.vidpath)
        self.camera_info = self.load_intrinsics()        
        # An object to run dope node
        self.dope_node = DopeNode(self.pose_config)

    def load_configs(self):
        with open(self.opt.config) as f:
            pose_config = yaml.load(f, Loader=yaml.FullLoader)
        return pose_config

    def setup_realsense(self):
        # Configure depth and color streams
        print("Realsense")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # Start streaming
        profile = pipeline.start(config)
        return pipeline,config,profile

    def create_output_folder(self):
        try:
            shutil.rmtree(f"{self.opt.outf}")
        except:
            pass
        try:
            os.makedirs(f"{self.opt.outf}")
        except OSError:
            pass
        print (f"output is located in {self.opt.outf}")

    def load_intrinsics(self):
        with open(self.opt.camera) as f:
            rs_camera_info = yaml.load(f, Loader=yaml.FullLoader)
        return rs_camera_info

    def get_rs_intrinsics(self):
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_info = {'camera_matrix' : {'data' : [intr.fx, 0, intr.ppx, 0,0, intr.fy, intr.ppy, 0,0,0,1 ,0]}}

    def get_frames(self):
        if self.opt.realsense:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            frame = np.asanyarray(color_frame.get_data())
        else:
            ret,frame = self.cap.read()
        frame = frame[...,::-1].copy()
        return frame

def find_object(frame,camera_info,dope_node,opt):
    start = time.time()

    # call the inference node
    loc,ori,img = dope_node.image_callback(
        frame, 
        camera_info,
        output_folder = opt.outf,
        resize = opt.resize
    )

    elapsed = time.time() - start

    return loc,ori,img,elapsed
    