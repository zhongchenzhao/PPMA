#!/bin/bash

# update pip
# python3 -m pip install --upgrade pip


# install requirements.txt
pip install -r requirements.txt


# install mmcv
pip install openmim==0.3.9;
mim install mmcv-full==1.7.2;
mim install mmcv==1.7.2;
pip install mmsegmentation==0.30.0;
pip install mmdet==2.28.2;
