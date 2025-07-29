FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3.11 pip

RUN mkdir -p /home/ae
WORKDIR /home/ae
COPY . Ark
WORKDIR /home/ae/Ark
RUN python3.11 -m pip install -r requirement_torch.txt
RUN python3.11 -m pip install -r requirement_benchmark.txt
RUN python3.11 -m pip install -r requirement.txt
RUN python3.11 -m pip install -e .
