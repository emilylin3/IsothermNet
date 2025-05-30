# hash:sha256:7ac36b4067ba0ca008d00cdb94b0ad31895547b4e08c2957e9e7165a679a53e9
FROM registry.codeocean.com/codeocean/pytorch:2.1.0-cuda11.8.0-mambaforge23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install -U --no-cache-dir \
    matplotlib==3.9.1 \
    networkx==3.2.1 \
    numpy==1.26.4 \
    openpyxl==3.1.5 \
    optuna==3.6.1 \
    pandas==2.2.2 \
    psutil==6.0.0 \
    pymatgen==2024.7.18 \
    rdkit==2024.3.3 \
    rdkit-pypi==2022.9.5 \
    scipy==1.13.1 \
    sklearn==0.0 \
    torch==2.4.0 \
    torch-geometric==2.5.3 \
    tqdm==4.66.4
