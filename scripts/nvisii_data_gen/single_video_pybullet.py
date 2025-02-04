import os
import cv2
import glob
import math
import time
import nvisii as visii
import numpy as np
import pybullet as p
import random
import argparse
import colorsys
import subprocess
from math import acos
from math import sqrt
from math import pi    

from utils import *

from multiprocessing import Pool
from math import ceil
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    '--spp', 
    default=10,
    type=int,
    help = "number of sample per pixel, higher the more costly"
)
parser.add_argument(
    '--width', 
    default=500,
    type=int,
    help = 'image output width'
)
parser.add_argument(
    '--height', 
    default=500,
    type=int,
    help = 'image output height'
)

parser.add_argument(
    '--obj',
    default = 'listerine',
    help= 'Object to train on'
)

# TODO: change for an array
parser.add_argument(
    '--objs_folder_distrators',
    default=f'../nvisii_data_gen/google_scanned_models/',
    help = "object to load folder"
)
parser.add_argument(
    '--objs_folder',
    default=f'../nvisii_data_gen/models/',
    help = "object to load folder"
)
parser.add_argument(
    '--skyboxes_folder',
    default=f'../nvisii_data_gen/dome_hdri_haven/',
    help = "dome light hdr"
)
parser.add_argument(
    '--nb_objects',
    default=str(int(random.uniform(50,75))),
    help = "how many objects"
)
parser.add_argument(
    '--nb_distractors',
    default=str(int(random.uniform(20,30))),
    help = "how many objects"
)
parser.add_argument(
    '--nb_frames',
    default=200,
    help = "how many frames to save"
)
parser.add_argument(
    '--skip_frame',
    default=100,
    type=int,
    help = "how many frames to skip"
)
parser.add_argument(
    '--noise',
    action='store_true',
    default=False,
    help = "if added the output of the ray tracing is not sent to optix's denoiser"
)
parser.add_argument(
    '--outf',
    default='dataset/',
    help = "output filename inside output/"
)
parser.add_argument('--seed',
    default = None, 
    help = 'seed for random selection'
)

parser.add_argument(
    '--interactive',
    action='store_true',
    default=False,
    help = "make the renderer in its window"
)

parser.add_argument(
    '--motion',
    action='store_true',
    default=False,
    help = "show motion images."
)

parser.add_argument(
    '--motionblur',
    action='store_true',
    default=False,
    help = "use motion blur to generate images"
)

parser.add_argument(
    '--static_camera',
    action='store_true',
    default=False,
    help = "make the camera static"
)

parser.add_argument(
    '--box_size',
    default=0.5,
    type=float,
    help = "make the object movement easier"
)

parser.add_argument(
    '--sage',
    action='store_true',
    default=False,
    help = "run on sagemaker"
)

parser.add_argument(
    '--upload',
    action='store_true',
    default=False,
    help = "upload sample imgs to s3"
)

parser.add_argument(
    '--generate_only',
    action='store_true',
    default=False,
    help = "Only generate images and upload to S3"
)


opt = parser.parse_args()

# # # # # # # # # # # # # # # # # # # # # # # # #
outp = './output/'
if opt.sage:
    outp = "/opt/ml/input/data/channel1"
    print(f"Output folder {outp}")
    if os.path.isdir(f'/opt/ml/checkpoints/{opt.outf}_{opt.spp}/'):
        print(f"Folder /opt/ml/checkpoints/{opt.outf}_{opt.spp}/ exists")
    else:
        os.makedirs(f'/opt/ml/checkpoints/{opt.outf}_{opt.spp}/')
        print(f'Created checkpoint folder /opt/ml/checkpoints/{opt.outf}_{opt.spp}/')

if os.path.isdir(outp):
    print(f'folder {outp}/ exists')
else:
    os.makedirs(outp)
    print(f'created folder {outp}/')

if os.path.isdir(f'{outp}/{opt.outf}_{opt.spp}'):
    print(f'folder {outp}/{opt.outf}_{opt.spp}/ exists')
else:
    os.makedirs(f'{outp}/{opt.outf}_{opt.spp}')
    print(f'Created output folder {outp}/{opt.outf}_{opt.spp}/')


outf = f'{opt.outf}_{opt.spp}'
opt.outf = f'{outp}/{outf}'
opt.checkpt = f'/opt/ml/checkpoints/{outf}'

if not opt.seed is None:
    random.seed(int(opt.seed)) 

# # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # #

def generate(frame_ids):
    visii.initialize(headless = not opt.interactive)

    if not opt.motionblur:
        visii.sample_time_interval((1,1))

    visii.sample_pixel_area(
        x_sample_interval = (.5,.5), 
        y_sample_interval = (.5, .5))

    # visii.set_max_bounce_depth(1)

    visii.enable_denoiser()


    camera = visii.entity.create(
        name = "camera",
        transform = visii.transform.create("camera"),
        camera = visii.camera.create_perspective_from_fov(
            name = "camera", 
            field_of_view = 0.785398, 
            aspect = float(opt.width)/float(opt.height)
        )
    )

    # data structure
    random_camera_movement = {
        'at':visii.vec3(1,0,0),
        'up':visii.vec3(0,0,1),
        'eye':visii.vec3(0,0,0)
    }

    camera.get_transform().look_at(
        at = random_camera_movement['at'], # look at (world coordinate)
        up = random_camera_movement['up'], # up vector
        eye = random_camera_movement['eye'],
    )

    visii.set_camera_entity(camera)

    # # # # # # # # # # # # # # # # # # # # # # # # #

    # lets turn off the ambiant lights 
    # load a random skybox 
    skyboxes = glob.glob(f'{opt.skyboxes_folder}/*.hdr')
    skybox_random_selection = skyboxes[random.randint(0,len(skyboxes)-1)]

    dome_tex = visii.texture.create_from_file('dome_tex',skybox_random_selection)
    visii.set_dome_light_texture(dome_tex)
    visii.set_dome_light_intensity(random.uniform(1.1,2))
    # visii.set_dome_light_intensity(1.15)
    visii.set_dome_light_rotation(
        # visii.angleAxis(visii.pi()/2,visii.vec3(1,0,0)) \
        # * visii.angleAxis(visii.pi()/2,visii.vec3(0,0,1))\
        visii.angleAxis(random.uniform(-visii.pi(),visii.pi()),visii.vec3(0,0,1))\
        # * visii.angleAxis(visii.pi()/2,visii.vec3(0,1,0))\
        # * visii.angleAxis(random.uniform(-visii.pi()/8,visii.pi()/8),visii.vec3(0,0,1))\
    )

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # Lets set some objects in the scene

    mesh_loaded = {}
    objects_to_move = []

    sample_space= [
            [-20,20],
            [-20,20],
            [-30,-2]
        ]

    if opt.interactive:
        physicsClient = p.connect(p.GUI) # non-graphical version
    else:
        physicsClient = p.connect(p.DIRECT) # non-graphical version

    visii_pybullet = []
    names_to_export = []

    def adding_mesh_object(name, obj_to_load,texture_to_load,scale=1, close=False):
        global mesh_loaded, visii_pybullet, names_to_export
        # obj_to_load = toy_to_load + "/meshes/model.obj"
        # texture_to_load = toy_to_load + "/materials/textures/texture.png"

        print("loading:",obj_to_load)

        if obj_to_load in mesh_loaded:
            toy_mesh = mesh_loaded[obj_to_load] 
        else:
            toy_mesh = visii.mesh.create_from_file(name,obj_to_load)
            mesh_loaded[obj_to_load] = toy_mesh

        toy = visii.entity.create(
            name = name,
            transform = visii.transform.create(name),
            mesh = toy_mesh,
            material = visii.material.create(name)
        )


        toy_rgb_tex = visii.texture.create_from_file(name,texture_to_load)
        toy.get_material().set_base_color_texture(toy_rgb_tex) 
        toy.get_material().set_roughness(random.uniform(0.1,0.5))

        toy.get_transform().set_scale(visii.vec3(scale))
        # If generate objects close to the camera only translate in x,y and not z
        if close: 
            x = random.uniform(-0.3,0.3)
            y = random.uniform(-0.3,0.3)
            z = 0
        else:
            x = random.uniform(0.1,2)
            y = random.uniform(-1,1)
            z = random.uniform(-1,1)
        # Scale the position based on depth into image to make sure it remains in frame
        #y *= x
        #z *= x

        #print(z,x,y)
        toy.get_transform().set_position(
            visii.vec3(
                x,
                y,
                z,
                )
            )

        toy.get_transform().set_rotation(
            visii.quat(
                random.uniform(0, 1),
                random.uniform(0, 1),
                random.uniform(0, 1),
                random.uniform(0, 1),
            )
        )

        id_pybullet = create_physics(name, mass = np.random.rand()*5)

        visii_pybullet.append(
            {
                'visii_id':name,
                'bullet_id':id_pybullet,
                'base_rot':None,
            }
        )
        gemPos, gemOrn = p.getBasePositionAndOrientation(id_pybullet)
        force_rand = 10
        object_position = 0.01
        p.applyExternalForce(
            id_pybullet,
            -1,
            [   random.uniform(-force_rand,force_rand),
                random.uniform(-force_rand,force_rand),
                random.uniform(-force_rand,force_rand)],
            [   random.uniform(-object_position,object_position),
                random.uniform(-object_position,object_position),
                random.uniform(-object_position,object_position)],
            flags=p.WORLD_FRAME
        )
        names_to_export.append(name)

        cuboid = add_cuboid(name, debug=False)

    google_content_folder = glob.glob(opt.objs_folder_distrators + "*")

    for i_obj in range(int(opt.nb_distractors)):

        toy_to_load = google_content_folder[random.randint(0,len(google_content_folder)-1)]

        obj_to_load = toy_to_load + "/meshes/model.obj"
        texture_to_load = toy_to_load + "/materials/textures/texture.png"
        name = "google_"+toy_to_load.split('/')[-2] + f"_{i_obj}"

        adding_mesh_object(name,obj_to_load,texture_to_load)


    google_content_folder = glob.glob(opt.objs_folder + "Artec/Artec")

    for i_obj in range(int(opt.nb_objects)):

        toy_to_load = google_content_folder[random.randint(0,len(google_content_folder)-1)]

        obj = opt.obj

        obj_to_load = toy_to_load + f"/{obj}/{obj}.obj"
        texture_to_load = toy_to_load + f"/{obj}/{obj}.jpg"
        name = f"{obj}" + toy_to_load.split('/')[-2] + f"_{i_obj}"
        adding_mesh_object(name,obj_to_load,texture_to_load,scale=1)

        # p.applyExternalTorque(id_pybullet,-1,
        #     [   random.uniform(-force_rand,force_rand),
        #         random.uniform(-force_rand,force_rand),
        #         random.uniform(-force_rand,force_rand)],
        #     [0,0,0],
        #     flags=p.WORLD_FRAME
        # )

    camera_pybullet_col = p.createCollisionShape(p.GEOM_SPHERE,0.05)
    camera_pybullet = p.createMultiBody(
        baseCollisionShapeIndex = camera_pybullet_col,
        basePosition = camera.get_transform().get_position(),
        # baseOrientation= rot,
    )    

    # print('simulate')
    # for i in range (10000):
    #     p.stepSimulation()
    #     # time.sleep(0.1)
    #     time.sleep(1./240.)

    box_position = opt.box_size

    plane1 = p.createCollisionShape(p.GEOM_PLANE,planeNormal = [0,0,-1])
    rot1 = visii.angleAxis(-visii.pi()/16,visii.vec3(0,1,0))
    plane1_body = p.createMultiBody(
        baseCollisionShapeIndex = plane1,
        basePosition = [0,0,-box_position],
        baseOrientation= [rot1[3],rot1[0],rot1[1],rot1[2]],
    )    

    plane1 = p.createCollisionShape(p.GEOM_PLANE,planeNormal = [0,0,1])
    rot1 = visii.angleAxis(visii.pi()/16,visii.vec3(0,1,0))
    plane2_body = p.createMultiBody(
        baseCollisionShapeIndex = plane1,
        basePosition = [0,0,box_position],
        baseOrientation= [rot1[3],rot1[0],rot1[1],rot1[2]],
    )    


    plane1 = p.createCollisionShape(p.GEOM_PLANE,planeNormal = [0,1,0])
    rot1 = visii.angleAxis(visii.pi()/16,visii.vec3(1,0,0))
    plane3_body = p.createMultiBody(
        baseCollisionShapeIndex = plane1,
        basePosition = [0,+box_position,0],
        baseOrientation= [rot1[3],rot1[0],rot1[1],rot1[2]],
    )    

    plane1 = p.createCollisionShape(p.GEOM_PLANE,planeNormal = [0,-1,0])
    rot1 = visii.angleAxis(-visii.pi()/16,visii.vec3(1,0,0))
    plane4_body = p.createMultiBody(
        baseCollisionShapeIndex = plane1,
        basePosition = [0,-box_position,0],
        baseOrientation= [rot1[3],rot1[0],rot1[1],rot1[2]],
    )    


    plane1 = p.createCollisionShape(p.GEOM_PLANE,planeNormal = [-1,0,0])
    plane5_body = p.createMultiBody(
        baseCollisionShapeIndex = plane1,
        basePosition = [2,0,0],
        # baseOrientation= [rot1[3],rot1[0],rot1[1],rot1[2]],
    )    

    plane1 = p.createCollisionShape(p.GEOM_PLANE,planeNormal = [1,0,0])
    plane6_body = p.createMultiBody(
        baseCollisionShapeIndex = plane1,
        basePosition = [0.1,0,0],
        # baseOrientation= [rot1[3],rot1[0],rot1[1],rot1[2]],
    )    


    # # # # # # # # # # # # # # # # # # # # # # # # #



    i_frame = -1
    i_render = 0
    condition = True
    print(f'spp : {opt.spp}, save path : {opt.outf}')

    for i in tqdm(frame_ids): 
        start = time.time()
        p.stepSimulation()
        ### add skip updates here.

        i_frame += 1  


        if i_render >= int(opt.nb_frames) and not opt.interactive:
            condition = False

        # update the position from pybullet
        for i_entry, entry in enumerate(visii_pybullet):
            # print('update',entry)
            visii.entity.get(entry['visii_id']).get_transform().set_position(
                visii.entity.get(entry['visii_id']).get_transform().get_position(),
                previous = True
            )
            visii.entity.get(entry['visii_id']).get_transform().set_rotation(
                visii.entity.get(entry['visii_id']).get_transform().get_rotation(),
                previous = True
            )

            update_pose(entry)
            if random.random() > 0.99:
                force_rand = 10
                object_position = 0.01
                p.applyExternalForce(
                    entry['bullet_id'],
                    -1,
                    [   random.uniform(-force_rand,force_rand),
                        random.uniform(-force_rand,force_rand),
                        random.uniform(-force_rand,force_rand)],
                    [   random.uniform(-object_position,object_position),
                        random.uniform(-object_position,object_position),
                        random.uniform(-object_position,object_position)],
                    flags=p.WORLD_FRAME
                )
        if i_frame == 0: 
            continue

        if not opt.interactive:
            
            if not i_frame % int(opt.skip_frame) == 0:
                continue
            
            i_render +=1

            visii.sample_pixel_area(
                x_sample_interval = (0,1), 
                y_sample_interval = (0,1))

            if opt.sage:
                # Only generate image, upload all to S3
                if opt.generate_only:
                    save_path = opt.checkpt
                else:
                    # Upload every first image to S3 for inspection which is used for training
                    if i_render == 1:
                        save_path = opt.checkpt
                    else:
                        save_path = opt.outf
            else:
                save_path = opt.outf
            

            print(f"{str(i_render).zfill(5)}/{str(opt.nb_frames).zfill(5)}")

            visii.render_to_file(
                width=int(opt.width), 
                height=int(opt.height), 
                samples_per_pixel=int(opt.spp),
                file_path=f"{save_path}/{str(i_render).zfill(5)}_{opt.spp}.png"    
            )
            visii.sample_pixel_area(
                x_sample_interval = (.5,.5), 
                y_sample_interval = (.5,.5))

            visii.sample_time_interval((1,1))

            visii.render_data_to_file(
                width=opt.width, 
                height=opt.height, 
                start_frame=0,
                frame_count=1,
                bounce=int(0),
                options="entity_id",
                file_path = f"{save_path}/{str(i_render).zfill(5)}_{opt.spp}.seg.exr"
            )
            segmentation_array = visii.render_data(
                width=int(opt.width), 
                height=int(opt.height), 
                start_frame=0,
                frame_count=1,
                bounce=int(0),
                options="entity_id",
            )        
            export_to_ndds_file(
                f"{save_path}/{str(i_render).zfill(5)}_{opt.spp}.json",
                obj_names = names_to_export,
                width = opt.width,
                height = opt.height,
                camera_name = 'camera',
                # cuboids = cuboids,
                camera_struct = random_camera_movement,
                segmentation_mask = np.array(segmentation_array).reshape(opt.width,opt.height,4)[:,:,0],
                visibility_percentage = False,
            )
            visii.render_data_to_file(
                width=opt.width, 
                height=opt.height, 
                start_frame=0,
                frame_count=1,
                bounce=int(0),
                options="depth",
                file_path = f"{save_path}/{str(i_render).zfill(5)}_{opt.spp}.depth.exr"
            )

            #print(f'Length of directory {save_path} is {len(os.listdir(save_path))}')

            print(f'Time taken {time.time() - start}')


    #subprocess.call(['ffmpeg', '-y',\
    #    '-framerate', '30', "-hide_banner", "-loglevel", \
    #    "panic",'-pattern_type', 'glob', '-i',\
    #    f"{opt.outf}/*.png", f"{opt.outf}/video.mp4"]) 

    visii.deinitialize()


# Start processes

# l = list(range(int(opt.nb_frames)))
# nb_process = 4 

# # How many frames each process should have 
# n = ceil(len(l) / nb_process)

# # divide frames between processes
# frame_groups = [l[i:i + n] for i in range(0, len(l), n)]

# with Pool(nb_process) as p:
#     p.map(generate,frame_groups)