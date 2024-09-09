# To push a new version, run:
# $ docker build -f cuda12.3.2-cudnn8.9-devel-ubuntu20.04.Dockerfile . --tag "tfra/nosla-cuda12.3.2-cudnn8.9-devel-ubuntu20.04"
# $ docker push tfra/nosla-12.3.2-cudnn8.9-devel-ubuntu20.04

# downlod https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/12.x/cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb/
# en
FROM nvidia/cuda:12.3.2-devel-ubuntu20.04 as devtoolset
RUN apt-get update
#RUN apt-get update && apt-cache search linux-headers

#RUN apt-get install linux-headers-$(uname -r)
#RUN apt-key del 7fa2af80
RUN rm -rf /usr/share/keyrings/cuda-archive-keyring.gpg
RUN apt-get -y install wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb

RUN dpkg -i cuda-keyring_1.1-1_all.deb  && rm cuda-keyring_1.1-1_all.deb
RUN cat /etc/apt/sources.list && \
    ls -l /etc/apt/sources.list.d/

RUN rm -f /etc/apt/sources.list.d/cuda*

RUN echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/cuda.list

RUN apt-get update && \
    apt-get -y install cudnn8-cuda-12

RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/*