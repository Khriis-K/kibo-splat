#!/bin/bash
# Download the two datasets. Compute nodes have no internet: run this on the transfer
# node, inside tmux:
#
#   ssh <user>@hpc-transfer1.usc.edu
#   tmux new -s data
#   cd <repo> && bash carc/download_data.sh
#
# 1. Mip-NeRF 360 `garden` (learning scene, COLMAP poses included). The zip holds all
#    7 scenes (several GB); we unpack one and delete the zip.
# 2. Astrobee ISS dataset, sequence `iva_kibo_trans` (the NASA demo; 2.0 GB, Google Drive).
#    1,068 grayscale 1280x880 frames with NASA ground-truth poses, TUM format.
#    Source: https://astrobee-iss-dataset.github.io (Kang et al., IEEE RA-L 2024).
set -euo pipefail
DATA="/scratch1/$USER/kibo-splat/data"
mkdir -p "$DATA" && cd "$DATA"

# --- 1. garden
if [ ! -d "360_v2/garden" ]; then
    wget -c http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
    mkdir -p 360_v2
    unzip -q 360_v2.zip "garden/*" -d 360_v2
    rm 360_v2.zip
fi
ls "360_v2/garden"                       # expect: images images_2 images_4 images_8 poses_bounds.npy sparse

# --- 2. Kibo
KIBO_ID="13t4wNxYJ1V6yaOvPMagM2jRffDUV-TUZ"   # iva_kibo_trans.zip (TUM format)
if [ ! -d "astrobee/iva_kibo_trans/gray" ]; then
    mkdir -p astrobee && cd astrobee
    curl -L -C - -o iva_kibo_trans.zip \
        "https://drive.usercontent.google.com/download?id=${KIBO_ID}&export=download&confirm=t"
    unzip -q iva_kibo_trans.zip -d iva_kibo_trans
    (cd iva_kibo_trans && unzip -q gray.zip && rm gray.zip gray_raw.zip)
    rm iva_kibo_trans.zip
    cd ..
fi
wc -l astrobee/iva_kibo_trans/groundtruth.txt   # expect ~1068 poses
ls astrobee/iva_kibo_trans/gray | wc -l         # expect 1068 images
