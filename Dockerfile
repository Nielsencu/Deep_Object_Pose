FROM nvcr.io/nvidia/pytorch:21.11-py3

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.
# To build:
# docker build -t nvidia-dope:noetic-v1 -f Dockerfile.noetic ..

ENV HOME /root
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /opt/ml/code \
    && mkdir -p /opt/ml/model

RUN mkdir -p /workspace/dope

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

# Copying local dataset
COPY scripts/nvisii_data_gen/output/dataset/ /opt/ml/input/data/channel1
COPY scripts/hyperparameters.json /opt/ml/input/config/

COPY . /workspace/dope

# Copy script for sagemaker to use
COPY scripts/train2/train.py /opt/ml/code/train.py  

ENV PYTHONPATH "${PYTHONPATH}:/workspace/dope/scripts/train2"
RUN echo $PYTHONPATH

ENV DISPLAY :0
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
ENV TERM=xterm

# Some QT-Apps don't show controls without this
ENV QT_X11_NO_MITSHM 1

ENV SAGEMAKER_PROGRAM train.py

# Specify train.py parameters
ENTRYPOINT ["python", "-m", "torch.distributed.launch", "/opt/ml/code/train.py"]