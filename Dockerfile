FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

ARG http_proxy=''
ARG https_proxy=''

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VERSION 6.0.20

RUN apt update
RUN apt install -y --allow-unauthenticated --no-install-recommends \
    build-essential cmake vim curl \
    libjpeg-dev libpng-dev   \
    libgtk3.0 libsm6 python3-venv cmake ffmpeg pkg-config \
    qtbase5-dev libqt5opengl5-dev libassimp-dev  \
    libboost-python-dev libtinyxml-dev bash python3-tk libcudnn6=$CUDNN_VERSION-1+cuda8.0 \
    libcudnn6-dev=$CUDNN_VERSION-1+cuda8.0 libosmesa6-dev \
    libopenmpi-dev libglew-dev

RUN apt install -y python3.5 python3-pip python3-setuptools libpython3.5-dev apt-utils ca-certificates software-properties-common wget unzip git

RUN pip3 install pip --upgrade --proxy $http_proxy''

RUN add-apt-repository ppa:jamesh/snap-support && apt-get update && apt install -y patchelf
RUN rm -rf /var/lib/apt/lists/*

# For some reason, I have to use a different account from the default one.
# This is absolutely optional and not recommended. You can remove them safely.
# But be sure to make corresponding changes to all the scripts.

WORKDIR /shaang
RUN chmod -R 777 /shaang && chmod -R 777 /usr/local && useradd -d /shaang -u 13071 shaang
USER shaang

RUN mkdir -p /shaang/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /shaang/.mujoco \
    && rm mujoco.zip
RUN wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /shaang/.mujoco \
    && rm mujoco.zip

ENV LD_LIBRARY_PATH /shaang/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /shaang/.mujoco/mjpro200_linux/bin:${LD_LIBRARY_PATH}

RUN git config --global http.proxy $http_proxy''

# Make sure you have a license, otherwise comment this line out
# Of course you then cannot use Mujoco and DM Control, but Roboschool is still available
COPY ./mjkey.txt /shaang/.mujoco/mjkey.txt

COPY requirements.txt requirements.txt
RUN pip --proxy $http_proxy'' install -r requirements.txt
RUN pip --proxy $http_proxy'' install git+https://github.com/openai/baselines.git@8e56dd#egg=baselines

# Additional stuff for udacity project:
RUN pip --proxy $http_proxy''  install unityagents
# This is reacher 1-agent headless variant
RUN wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip -P /shaang/DeepRL
RUN unzip /shaang/DeepRL/Reacher_Linux_NoVis.zip -d /shaang/DeepRL && mv /shaang/DeepRL/Reacher_Linux_NoVis /shaang/DeepRL/Reacher_Linux

# Force-install newer protobuf version, even if unityagents wants protobuf==3.4.0
# Needed for gym code to run, and looks like it's working with unity as well
RUN pip --proxy $http_proxy'' install "protobuf==3.9.1"

USER root

# Copy code
COPY ./deep_rl /shaang/DeepRL/deep_rl
COPY examples.py /shaang/DeepRL/
RUN chown -R shaang /shaang && chgrp -R shaang /shaang
# Our cluster runs containers as a different user
RUN chmod -R a+rwx /shaang/DeepRL/

USER root

WORKDIR /shaang/DeepRL

# ENTRYPOINT ["python3", "examples.py"]
