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

  generation_strategy = {'type': 'beam', 'beam_size': 1}

  # image_path = "/home/anh/src/caffe_lrcn/images/bellpeppers.jpg"
  image_path = "/home/anh/src/caffe_lrcn/images/brambling.jpg"

  descriptors = captioner.compute_descriptors([image_path])

  # Generate captions for all images.
  temp = float('inf')
  output_captions, output_probs = captioner.sample_captions(
        descriptors[0], temp=temp)

  #print ">>> output_captions", output_captions
  caption = output_captions[0]
  words_caption = captioner.sentence(caption)
  print ">>>> : [", words_caption, "]"

  # experimenter = CaptionExperiment(captioner, image_path=image_path)
  # experimenter.generation_experiment()

if __name__ == "__main__":
  main()
