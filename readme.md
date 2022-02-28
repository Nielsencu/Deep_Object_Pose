# DOPE Repository
This repository contains the complete DOPE scripts to generate synthetic data, train, and run inference. 

Robot Picking             |  Robot picking with orientation
:-------------------------:|:-------------------------:|:-------------------------:
![Robot gripping](doc/current_progress.mp4?raw=true "Robot gripping")  |  ![Robot picking with orientation](doc/pick_with_orientation.mp4 "Robot picking with orientation") 

## Synthetic Data Generation
To achieve sim-to-real learning, synthetic data generated is made to be as real as possible to mimic real-world inference. Firstly, it places distractors in the scene, which are 3D meshes of random objects placed randomly in the scene. Placing a bunch of random distractors teaches the model to learn to differentiate the object we are trying to train on and other "distractors" which can be random objects in the background when doing inference in the real world. Secondly, it uses HDR lighting to render images with realistic lighting. Thirdly, realistic random backgrounds will be overlayed for every scene generated.

Aveeno with 100 sample-per-pixel             |  Listerine with 2000 sample-per-pixel |  Blue Baby Wipes with random distractors
:-------------------------:|:-------------------------:|:-------------------------:
![Aveeno 100 spp](doc/aveeno_100.png?raw=true "Aveeno 100 spp")  |  ![Listerine 2000 spp](doc/listerine_2000.png?raw=true "Listerine 2000 spp") | ![Babywipes with distractors](doc/babywipes_with_distractors.png?raw=true "Baby wipes with distractors")


The dockerfile for image generation can be found in Dockerfile.datasage. This is the dockerfile that generates synthetic data generator docker image, which is available in AWS ECR. To run data generation with AWS Sagemaker (sample run), there are configurations for image/data generator:
* spp: sampler-per-pixel determines your image resolution, higher spp results in a longer image generation time
* nb_frames: number of frames for every scene generated, if nb_frames = 200, then for every new scene (new background and HDR) it will capture 200 images of that scene which has the same background, HDR, 3D objects and distractors. The higher nb_frames, the less training time because it uses the same scene and doesn't have to place new objects in the scene. Since forces and torques are applied to each 3D objects, every time frames are captured, different position/orientation of the 3D object will also be captured.
* nb_distractors: number of distractors to be placed in the scene ([50,75] means it will generate 50-75 distractors every scene). The higher number of distractors, the longer it takes to load a new scene.
* nb_objects: number of objects placed in the scene ([20,30] means it will generate 20-30 distractors every scene). The higher number of objects, the longer it takes to load a new scene.
* obj: the name of your object (3D CAD model and its texture should be present in S3 bucket s3://jiazheng-hd/dope_datagen/models/obj/obj). Note: Usually 3D models scanned from 3D Scanners such as one from Artec needs to be decimated before it can be rendered using this graphics engine, I usually decimate till it has 15000 triangles.

The actual synthetic data generation is inside `scripts/nvisii_data_gen/single_video_pybullet.py`. To adjust the distance of the object relative to the camera, it can be done inside `single_video_pybullet.py`. The script `scripts/generate_data.py` will call `single_video_pyubllet.py` and also parse, and pass in the configurations through a json file located in `/opt/ml/input/config/hyperparameters.json` when run in AWS Sagemaker.

To see the random distractors that are used, it is located in s3://jiazheng-hd/dope_datagen/google_scanned_models. HDR lightings can also be found in s3://jiazheng-hd/dope_datagen/dome_hdri_haven/.

I have a separate synthetic data generator docker image that can be run on AWS Sagemaker. It uses a cheaper training instance ml.g4dn.2xlarge $1/hour. In AWS ECR, there are two synthetic data generator, one is imagegen-close and one is imagegen. The imagegen-close shouldn't be used as I used it to experiment whether if objects placed closer to camera would yield a better model accuracy. However, I found that placing objects closer to camera means that less objects can be viewed in a single image. On the other hand, placing objects further away from camera allows more objects to be viewed in a single image although the details might not be as clear as when it is closer to camera. My findings show that the 'far' model still outperforms the 'close' model as the 'close' model's accuracy was poorer although both had been trained on the same 20000 images (video can be found in AI slack channel).

https://ap-southeast-1.console.aws.amazon.com/sagemaker/home?region=ap-southeast-1#/jobs/dope-imagegenclose-bluebabywipes

## Training
Model training can be done using either single GPU or multi-GPU in AWS Sagemaker.

For single GPU training sample run refer to ([sample run](https://ap-southeast-1.console.aws.amazon.com/sagemaker/home?region=ap-southeast-1#/jobs/dope-training-singlegpu-48-batchsize-bluebabywipes-2)). The input data config consists of the weights folder, datagen folder, and channel1 folder which should point to the training data. Output data config folder will save all the training weights in .pth format. Some important hyperparameters include:

* batch_size : Training batch size (set default to 48 as it is the highest you can go)
* epochs : number of epochs to run (set default to 60 for training a new object)
* generator : if set to 0, it will skip synthetic data generation and use the channel1 folder as the training data. If set to 1, it will generate synthetic data before the training and commence the training as soon as synthetic data generation completes.
* gpus: gpu ids that are used, single gpu set to 0, four gpus set to 0 1 2 3
* imgs : you need to change this if you set generator to 1, as this specifies the number of images you want to generate for training
* nb_frames : passed to synth data gen
* net : if set to 0, it will use DOPE's pretrained weight, else put as pretrained weight file name e.g `net_epoch_60.pth` the weight should exist inside s3://jiazheng-hd/dope_models/.
* obj : passed to synth data gen
* optimizer : training optimizer (set default to adam)
* sage : always set to 1 if trained on sagemaker else it will use local directories to find input data such as 3D models, weights, and so on
* spp : passed to synth data gen
* subbatch_size : if batch_size == subbatch_size, it will run normal mini-batch gradient descent, else e.g if batch size 128 and subbatch size is 32, it will process 4 subbatches, accumulate the gradients and update the network using the accumulated gradients. This is done so that higher effective batch size can be employed (refer to [this](https://github.com/NVlabs/Deep_Object_Pose/issues/155)). Apparently this makes the loss to converge faster, however, from my own testing, it doesn't seem to be the case and the loss seems to just fluctuate hence just use batch_size == subbatch_size. 
* workers: number of Pytorch data loader workers. 8 for single GPU, 12 for multi GPU

For multi GPU training sample run refer to ([sample run](https://ap-southeast-1.console.aws.amazon.com/sagemaker/home?region=ap-southeast-1#/jobs/dope-training-test-fourgpus-48-batchsize-listerine-1)). It isn't recommended to run synthetic data generation together with the multi GPU training because data generation isn't optimized with multi GPU and multi GPU instance is expensive so that's why I have an image generator that uses a cheap instance to perform image generation, and pass the S3 folder output of the image generation as channel1 input for multi GPU training. For multi GPU training, important hyperparameters that needs to be changed are epochs and net (optional ones include batch_size, subbatch_size, optimizer, workers).

I have implemented training with checkpoints, so it can utilize AWS Spot instance training with 70% savings. You need to create new checkpoints S3 folder for every new training session and pass it as S3 output path in Checkpoint Configuration part in AWS Sagemaker. It will automatically upload latest training weights to the cloud.

You can also monitor the training loss of the algorithm in the view algorithm metrics to see how your training performs. By 30 epochs, there should be a downward trend, otherwise if it is just fluctuating for 30 epochs, it might signal that the model isn't able to learn much and you can do early termination.

For a more complete explanation, original repo can be found [here](https://github.com/NVlabs/Deep_Object_Pose) of how the training works.

## Scripts
To build and push your desired docker image to AWS ECR, you can change your dockerfile name in `sagemaker_docker/ecr_push.sh` and also AWS login credentials. The script `sagemaker_docker/run_dope_docker.sh` also allows you to run the docker image locally if you wish to. It uses NVIDIA docker run commands so that it can run NVIDIA GPU-based applications seamlessly.
## Further improvements:
- [ ] Currently, the synthetic data generation is done through NVISII Python Renderer. However, the data generation using NVISII takes a long time. To generate 50000 images it takes 3 days with the following configurations: 100 spp, 50-75 distractors, 20-30 objects, 200 nb_frames. It can be reduced further by reducing the amount of distractors and objects generated for each scene, but I have not experimented by how much it will affect the model accuracy. 100 spp and 200 nb_frames seems to be the most ideal and these two are the two configs that I had been experimenting with.
- [ ] Expensive to run image generation in AWS Sagemaker. Might be better to run with unity.
- [ ] Haven't tested multi-GPU's model accuracy. However, benchmarking shows 3x boost in training time using four vs single GPU.
- [ ] 