# To push a new version, run:
# $ docker build -f cuda12.3.2-cudnn8.9-devel-ubuntu20.04.Dockerfile . --tag "tfra/nosla-cuda12.3.2-cudnn8.9-devel-ubuntu20.04"
# $ docker login -u user -p token
# $ docker push tfra/nosla-cuda12.3.2-cudnn8.9-devel-ubuntu20.04
# https://developer.nvidia.com/rdp/cudnn-archive to find correct cudnn.
# https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/12.x/cudnn-local-repo-ubuntu2004-8.9.7.29_1.0-1_amd64.deb/
# dowloand to local and copy to the docker build context.
FROM nvidia/cuda:12.3.2-devel-ubuntu20.04 AS devtoolset
RUN apt-get update
RUN apt-get -y install wget
COPY cudnn-local-repo-ubuntu2004-8.9.7.29_1.0-1_amd64.deb /tmp/

RUN dpkg -i /tmp/cudnn-local-repo-ubuntu2004-8.9.7.29_1.0-1_amd64.deb

RUN cp /var/cudnn-local-repo-ubuntu2004-8.9.7.29/cudnn-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install /var/cudnn-local-repo-ubuntu2004-8.9.7.29/libcudnn*.deb
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/cudnn-local-repo-ubuntu2004-8.9.7.29/*.deb