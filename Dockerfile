FROM ubuntu:22.04
ENV PATH="/root/miniconda3/bin:$PATH"
ARG PATH="/root/miniconda3/bin:$PATH"

RUN apt-get update
RUN apt-get --fix-missing -y install graphviz libngspice0
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda create -n ark python=3.10 -y 
SHELL ["conda", "run", "-n", "ark", "/bin/bash", "-c"]
RUN conda install pip
RUN mkdir -p /home/ae
WORKDIR /home/ae
COPY . Ark
RUN cd Ark && pip install install -e .
RUN pysmt-install --z3 --confirm-agreement
RUN conda install -c conda-forge pyspice
RUN conda init bash

