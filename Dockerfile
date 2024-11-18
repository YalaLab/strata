FROM nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu22.04
ARG http_proxy
ARG https_proxy
ARG no_proxy

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}
ADD . /root/strata
WORKDIR /root/strata
RUN apt update && apt install -y htop python3.10 python3-pip git python-is-python3 vim
RUN /root/strata/setup_docker.sh

CMD ["/bin/bash"]
