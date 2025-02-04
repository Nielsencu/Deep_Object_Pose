FROM nvcr.io/nvidia/pytorch:21.11-py3

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.
# To build:
# docker build -t nvidia-dope:noetic-v1 -f Dockerfile.noetic ..

ENV HOME /root
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /opt/ml/code \
    && mkdir -p /opt/ml/model \
    && mkdir -p /opt/ml/input/data/channel1 \
    && mkdir -p /opt/ml/input/data/weights \
    && mkdir -p /opt/ml/input/data/datagen/models \
    && mkdir -p /opt/ml/input/data/datagen/dome_hdri_haven \
    && mkdir -p /opt/ml/input/data/datagen/google_scanned_models \
    && mkdir -p /workspace/dope

# Install system and development components
RUN apt-get update && apt-get -y --no-install-recommends install \
    apt-utils \
    software-properties-common \
    build-essential \
    cmake \
    git \
    python3-pip \
    libxext6 \
    libx11-6 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    freeglut3-dev \
    && apt-get -y autoremove \
    && apt-get clean

# pip install required Python packages
COPY requirements.txt ${HOME}
COPY scripts/train2/requirements1.txt ${HOME}
RUN python3 -m pip install --no-cache-dir -r ${HOME}/requirements.txt
RUN python3 -m pip install --no-cache-dir -r ${HOME}/requirements1.txt 

# Copy codes locally
COPY scripts /workspace/dope/scripts
COPY src /workspace/dope/src

# By git clone
# git clone git@github.com:NVlabs/Deep_Object_Pose.git /workspace/dope

# Copy script for sagemaker to use
COPY scripts/train2/generate_train.py /opt/ml/code/train.py

# If try on EC2
COPY scripts/hyperparameters.json /opt/ml/input/config/
# COPY scripts/train2/output/dataset/ /opt/ml/input/data/channel1
# COPY net_epoch_60.pth /opt/ml/input/data/weights/
# COPY scripts/nvisii_data_gen/models /opt/ml/input/data/datagen/models
# COPY scripts/nvisii_data_gen/google_scanned_models /opt/ml/input/data/datagen/google_scanned_models
# COPY scripts/nvisii_data_gen/dome_hdri_haven /opt/ml/input/data/datagen/dome_hdri_haven

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PYTHONPATH "${PYTHONPATH}:/workspace/dope/scripts/train2"
RUN echo $PYTHONPATH

ENV DISPLAY :0
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
ENV TERM=xterm

# Some QT-Apps don't show controls without this
ENV QT_X11_NO_MITSHM 1

# Specify train.py parameters
ENTRYPOINT ["torchrun", "--nproc_per_node=1", "--nnodes=1", "/opt/ml/code/train.py"]
