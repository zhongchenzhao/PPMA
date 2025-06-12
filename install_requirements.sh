#!/bin/bash

# update pip
# python3 -m pip install --upgrade pip


# install requirements.txt
pip install -r requirements.txt


# install mmcv
pip install openmim;
mim install mmcv-full;
mim install "mmcv < 2.0.0";
pip install mmsegmentation==0.30.0;
pip install mmdet==2.28.2;
