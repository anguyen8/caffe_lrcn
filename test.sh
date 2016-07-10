#!/bin/bash

img="${1}"

ver=7.0
export LD_LIBRARY_PATH=/home/anh/local/lib:/home/anh/160503__cudnn/cuda-${ver}/targets/x86_64-linux/lib:/home/anh/anaconda/lib:${LD_LIBRARY_PATH}

rm -rf retrieval_cache/
python examples/coco_caption/retrieval_test.py ${img}
