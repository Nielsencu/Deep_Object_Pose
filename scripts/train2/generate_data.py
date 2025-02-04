import random 
import subprocess
import time

import argparse
import os
import random
import shutil
try: 
    import configparser as configparser
except ImportError: 
    import ConfigParser as configparser

import datetime
import json

import warnings
warnings.filterwarnings("ignore")

print ("start:" , datetime.datetime.now().time())
start = time.time()

conf_parser = argparse.ArgumentParser(
    description=__doc__, # printed with -h/--help
    # Don't mess with format of description
    formatter_class=argparse.RawDescriptionHelpFormatter,
    # Turn off help, so we print all options in response to -h
    add_help=False
    )
conf_parser.add_argument("-c", "--config",
                        help="Specify config file", metavar="FILE")

parser = argparse.ArgumentParser()
parser.add_argument('train', default='train', help="train?")

# Read the config but do not overwrite the args written 
args, remaining_argv = conf_parser.parse_known_args()
defaults = { "option":"default" }
if args.config:
    config = configparser.SafeConfigParser()
    config.read([args.config])
    defaults.update(dict(config.items("defaults")))

parser.set_defaults(**defaults)
parser.add_argument("--option")
opt = parser.parse_args(remaining_argv)


hyperparameters_file = "/opt/ml/input/config/hyperparameters.json"
#hyperparameters_file = "../hyperparameters_generate.json"
# hyperparameters
with open(hyperparameters_file) as f:
   hyperparameters = json.load(f)
obj = hyperparameters["obj"]
imgs = int(hyperparameters["imgs"])
opt.spp = hyperparameters["spp"]
opt.sage = int(hyperparameters["sage"])
opt.nb_frames = hyperparameters["nb_frames"]
opt.nb_objects = hyperparameters["nb_objects"]
opt.nb_distractors = hyperparameters["nb_distractors"]
opt.generate_only = int(hyperparameters["generate_only"])
opt.generator = int(hyperparameters["generator"])

print(f"Generating {obj} with {imgs} images with {opt.spp} spp")

if opt.sage:
    print(f"Using sagemaker directories --------------------------------------------------------------------------")
    opt.data = ["/opt/ml/input/data/channel1"]
    opt.checkpt = "/opt/ml/checkpoints"
    data_gen_root = "/workspace/dope/scripts/nvisii_data_gen"
else:
    print(f"Using local directories --------------------------------------------------------------------------")
    # Use default
    opt.data = './output'
    opt.checkpt = ''
    data_gen_root = "../nvisii_data_gen"

num_loop = imgs // int(opt.nb_frames) # num of images = num_loop * nb_frames

print(f"Number of loops {num_loop}")

is_range_nb_objects = False
is_range_nb_distractors = False
if len(opt.nb_objects) > 1:
    is_range_nb_objects = True
    opt.nb_objects = opt.nb_objects.split(",")

if len(opt.nb_distractors) > 1:
    is_range_nb_distractors = True
    opt.nb_distractors = opt.nb_distractors.split(",")

# Synthetic data generation
if opt.generator:
    for i in range(0,num_loop):
        to_call = [
            "python",f'{data_gen_root}/single_video_pybullet.py',
            '--spp',f'{opt.spp}',
            '--nb_frames', f'{opt.nb_frames}',
            '--static_camera',
            '--outf',f"dataset/{str(i).zfill(3)}",
            '--obj', f'{obj}',
        ]
        to_call.append('--nb_objects')
        if is_range_nb_objects:
            to_call.append(f'{int(random.uniform(int(opt.nb_objects[0]), int(opt.nb_objects[1])))}')
        else:
            to_call.append(f'{opt.nb_objects}')
        to_call.append('--nb_distractors')
        if is_range_nb_distractors:
            to_call.append(str(int(random.uniform(int(opt.nb_distractors[0]), int(opt.nb_distractors[1])))))
        else:
            to_call.append(f'{opt.nb_distractors}')
        
        if opt.sage:
            to_call.append("--sage")
            datagen_folder = "/opt/ml/input/data/datagen"
            opt.objs_folder = datagen_folder + "/models/"
            print(f'Objects folder {opt.objs_folder}')
            to_call.append('--objs_folder')
            to_call.append(f'{opt.objs_folder}')
            opt.objs_folder_distrators = datagen_folder + "/google_scanned_models/"
            print(f'Distractors folder {opt.objs_folder_distrators}')
            to_call.append('--objs_folder_distrators')
            to_call.append(f'{opt.objs_folder_distrators}')
            opt.skyboxes_folder = datagen_folder + "/dome_hdri_haven/"
            print(f'Skyboxes folder {opt.skyboxes_folder}')
            to_call.append('--skyboxes_folder')
            to_call.append(f'{opt.skyboxes_folder}')
        if opt.generate_only:
            to_call.append('--generate_only')
        subprocess.call(to_call)
else:
    print("Skipping synthetic data generation")
    
print(f"Data generation took {time.time() - start} seconds")