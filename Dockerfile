FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
LABEL maintainer="alih@uoregon.edu"
RUN apt-get -y update \
&& apt-get install -y software-properties-common \
&& apt-get -y update \
&& add-apt-repository universe
COPY . /cct
WORKDIR /cct
