#!/bin/bash

. ~/workspace/bash-scripts/profiles/bash_profile

img="${1}"

if [ "$#" -ne "1" ]; then
    echo "Provide path to image"
    exit 1
fi
output_dir="output_caption"

ver=7.0
export LD_LIBRARY_PATH=/home/anh/local/lib:/home/anh/160503__cudnn/cuda-${ver}/targets/x86_64-linux/lib:/home/anh/anaconda/lib:${LD_LIBRARY_PATH}

rm -rf retrieval_cache/

mkdir -p ${output_dir}

caption=`python examples/coco_caption/make_caption.py ${img} | tail -1`
echo ${caption}

filename=$(basename "${img}")
output_file=${output_dir}/${filename}

cp ${img} ${output_file}
resize ${output_file} 256x256
add_label ${output_file} "${caption}" 20

echo "Saved to ${output_file}"
readlink -f ${output_file}
