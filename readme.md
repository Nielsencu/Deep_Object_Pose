# DOPE Repository
This repository contains the complete DOPE scripts to generate synthetic data, train, and run inference. I will a

## Synthetic Data Generation
To achieve sim-to-real learning, synthetic data generated is made to be as real as possible to mimic real-world inference. Firstly, it places distractors in the scene, which are 3D meshes of random objects placed randomly in the scene. This way, it can learn to differentiate the object we are trying to train on and other "distractors" which can be random objects in the background when doing inference in the real world. Secondly, it uses HDR lighting to render images with realistic lighting. Thirdly, the python renderer will generate realistic random backgrounds for every scene generated.

The dockerfile for image generation can be found in Dockerfile.datasage. This is the dockerfile that generates the synthetic image generator docker image, which is available in AWS ECR. To run image generation with AWS Sagemaker, there are configurations for image generator:
* spp:
* 

## Training
Training of the model

For a more complete explanation, original repo can be found [here](https://github.com/NVlabs/Deep_Object_Pose).

## Configurations
Configurations for model inference are inside the folder `config_inference`. 

For `config_pose.yaml`, important configurations are:
* `downscale_height`: 

## Further improvements:
- [ ] Currently, the synthetic data generation is done through NVISII Python Renderer. However, the data generation using NVISII takes a long time (50000 images takes 3 days).
- [ ] 