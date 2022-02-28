# DOPE Repository
This repository contains the complete DOPE scripts to generate synthetic data, train, and run inference. 

## Synthetic Data Generation
To achieve sim-to-real learning, synthetic data generated is made to be as real as possible to mimic real-world inference. Firstly, it places distractors in the scene, which are 3D meshes of random objects placed randomly in the scene. Placing a bunch of random distractors teaches the model to learn to differentiate the object we are trying to train on and other "distractors" which can be random objects in the background when doing inference in the real world. Secondly, it uses HDR lighting to render images with realistic lighting. Thirdly, realistic random backgrounds will be overlayed for every scene generated.

Aveeno with 100 sample-per-pixel             |  Listerine with 2000 sample-per-pixel |  Blue Baby Wipes with random distractors
:-------------------------:|:-------------------------:|:-------------------------:
![Aveeno 100 spp](doc/aveeno_100.png?raw=true "Aveeno 100 spp")  |  ![Listerine 2000 spp](doc/listerine_2000.png?raw=true "Listerine 2000 spp") | ![Babywipes with distractors](doc/babywipes_with_distractors.png?raw=true "Baby wipes with distractors")


The dockerfile for image generation can be found in Dockerfile.datasage. This is the dockerfile that generates synthetic image generator docker image, which is available in AWS ECR. To run image generation with AWS Sagemaker (sample ), there are configurations for image generator:
* spp:
* nb_distractors:
* nb_frames:
* nb_objects:
* obj:

There's an option to run image generation and training at the same time using the single GPU docker image

## Training
Training of the model

For a more complete explanation, original repo can be found [here](https://github.com/NVlabs/Deep_Object_Pose).

## Further improvements:
- [ ] Currently, the synthetic data generation is done through NVISII Python Renderer. However, the data generation using NVISII takes a long time (50000 images takes 3 days).
- [ ] 