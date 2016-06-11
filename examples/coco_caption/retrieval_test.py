#!/usr/bin/env python
import os
os.environ['GLOG_minloglevel'] = '2'  # suprress Caffe verbose prints

from collections import OrderedDict
import json
import numpy as np
import pprint
import cPickle as pickle
import string
import sys

# seed the RNG so we evaluate on the same subset each time
np.random.seed(seed=0)

# from coco_to_hdf5_data import *
from test_captioner import Captioner

def main():
  MAX_IMAGES = 1  # -1 to use all images
  
  ITER = 110000
  MODEL_FILENAME = 'lrcn_caffenet_iter_%d' % ITER
  DATASET_NAME = 'val'

  MODEL_DIR = "/raid/anh/from_jeffdonahue_lrcn"
  MODEL_FILE = '%s/%s.caffemodel' % (MODEL_DIR, MODEL_FILENAME)
  IMAGE_NET_FILE = './models/bvlc_reference_caffenet/deploy.prototxt'
  LSTM_NET_FILE = './examples/coco_caption/lrcn_word_to_preds.deploy.prototxt'
  VOCAB_FILE = './examples/coco_caption/h5_data/buffer_100/vocabulary.txt'
  DEVICE_ID = 0
  with open(VOCAB_FILE, 'r') as vocab_file:
    vocab = [line.strip() for line in vocab_file.readlines()]
  
  if MAX_IMAGES < 0: MAX_IMAGES = len(dataset.keys())
  captioner = Captioner(MODEL_FILE, IMAGE_NET_FILE, LSTM_NET_FILE, VOCAB_FILE,
                        device_id=DEVICE_ID)

  # image_path = "/home/anh/src/caffe_lrcn/images/bellpeppers.jpg"
  image_path = str(sys.argv[1]) #"/home/anh/src/caffe_lrcn/images/brambling.jpg"

  descriptor = captioner.compute_descriptors([image_path]).flat  # Get 1000 fc8 numbers from (1,1000) tensor

  # Generate captions for all images.
  temp = float('inf')
  # output_captions, _ = captioner.sample_captions( descriptor, temp=temp )
  # caption = output_captions[0]
  strategy = { "type": "sample", "temp": float("inf")}
  caption, _ = captioner.sample_caption(descriptor, strategy=strategy)

  print ">>> output_captions", caption
  
  words_caption = captioner.sentence(caption)
  print ">>>> : [", words_caption, "]"

if __name__ == "__main__":
  main()
