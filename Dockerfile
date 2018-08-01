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
    pip3 install --no-cache-dir Cython numpy pytest scipy opencv-python

COPY hdrnet/requirements.txt /opt/requirements.txt
RUN pip3 install --no-cache-dir -r /opt/requirements.txt
COPY hdrnet/ops/ /opt/hdrnet/ops
COPY hdrnet/Makefile /opt/hdrnet/
WORKDIR /opt/hdrnet
RUN make
ENV PYTHONPATH=/opt PATH="$PATH:/opt/hdrnet/bin"
COPY hdrnet /opt/hdrnet
