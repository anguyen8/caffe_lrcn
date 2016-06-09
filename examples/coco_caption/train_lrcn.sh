#!/usr/bin/env bash
ver=7.0
#export LD_LIBRARY_PATH=/home/anh/local/lib:/usr/local/cuda-${ver}/targets/x86_64-linux/lib:/home/anh/anaconda/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/home/anh/local/lib:/home/anh/160503__cudnn/cuda-${ver}/targets/x86_64-linux/lib:/home/anh/anaconda/lib:${LD_LIBRARY_PATH}

GPU_ID="3"
WEIGHTS=\
./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
DATA_DIR=./examples/coco_caption/h5_data/
if [ ! -d $DATA_DIR ]; then
    echo "Data directory not found: $DATA_DIR"
    echo "First, download the COCO dataset (follow instructions in data/coco)"
    echo "Then, run ./examples/coco_caption/coco_to_hdf5_data.py to create the Caffe input data"
    exit 1
fi

./build/tools/caffe train \
    -solver ./examples/coco_caption/lrcn_solver.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
