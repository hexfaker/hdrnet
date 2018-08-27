FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER hexfaker

RUN apt-get update && \
    apt-get install -y --no-install-recommends \ 
            python3 \
            python3-pip \
            python3-setuptools \
            python3-wheel \
            python3-magic \
            python3.5-dev \
            python-opencv \
            libtiff-dev \
            libpng-dev \
            libjpeg-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir Cython numpy pytest scipy opencv-python && \
    pip3 install --no-cache-dir http://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-linux_x86_64.whl

COPY hdrnet/requirements.txt /opt/requirements.txt
RUN pip3 install --no-cache-dir -r /opt/requirements.txt
COPY hdrnet/ops/ /opt/hdrnet/ops
COPY hdrnet/Makefile /opt/hdrnet/
WORKDIR /opt/hdrnet
RUN make
ENV PYTHONPATH=/opt:/opt/hdrnet PATH="$PATH:/opt/hdrnet/bin" OPENCV_OPENCL_DEVICE=disabled
COPY hdrnet /opt/hdrnet
