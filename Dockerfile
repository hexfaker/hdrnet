FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER hexfaker

RUN apt-get update && \
    apt-get install -y --no-install-recommends \ 
            python \
            python-pip \
            python-magic \
            python-setuptools \
            python2.7-dev \
            python-opencv \
            libtiff-dev \
            libpng-dev \
            libjpeg-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir Cython numpy pytest scipy

COPY hdrnet/requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
WORKDIR /opt/hdrnet
COPY hdrnet /opt/hdrnet
RUN make
