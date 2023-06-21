# Dockerfile to build a serving with TFRA OPs.
#
# To push a new version, run:
# $ docker build -f serving-devel.Dockerfile . --tag "tfra/serving:2.8.3-devel-gpu"
# $ docker push tfra/serving:2.8.3-devel-gpu

FROM tensorflow/serving:2.6.3-devel-gpu

RUN pip install --upgrade pip && pip install tensorflow==2.8.3

