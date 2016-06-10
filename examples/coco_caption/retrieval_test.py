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

# COCO_EVAL_PATH = './data/coco/coco-caption-eval'
# sys.path.append(COCO_EVAL_PATH)
# from pycocoevalcap.eval import COCOEvalCap

class CaptionExperiment():
  # captioner is an initialized Captioner (captioner.py)
  # dataset is a dict: image path -> [caption1, caption2, ...]
  def __init__(self, captioner, image_path):
    self.captioner = captioner
    
    self.image = image_path

  def compute_descriptors(self):
    self.descriptors = self.captioner.compute_descriptors([self.image])

  def generation_experiment(self, strategy, max_batch_size=1000):
    # Compute image descriptors.
    print 'Computing image descriptors'
    self.compute_descriptors()

    # Generate captions for all images.
    temp = float('inf')
    output_captions, output_probs = self.captioner.sample_captions(
          self.descriptors[0], temp=temp)

    #print ">>> output_captions", output_captions
    caption = output_captions[0]

    # Collect model/reference captions, formatting the model's captions and
    # each set of reference captions as a list of len(self.images) strings.
    # For each image, write out the highest probability caption.
    words_caption = self.captioner.sentence(caption)
    print ">>>> : [", words_caption, "]"


def main():
  MAX_IMAGES = 1  # -1 to use all images
  
  #ITER = 100000
  ITER = 110000
  #MODEL_FILENAME = 'lrcn_vgg_iter_%d' % ITER
  #MODEL_FILENAME = 'lrcn_caffenet_finetune_iter_%d' % ITER
  MODEL_FILENAME = 'lrcn_caffenet_iter_%d' % ITER
  DATASET_NAME = 'val'

  #MODEL_DIR = './examples/coco_caption'
  MODEL_DIR = "/raid/anh/from_jeffdonahue_lrcn"
  MODEL_FILE = '%s/%s.caffemodel' % (MODEL_DIR, MODEL_FILENAME)
  #MODEL_FILE = '%s/%s.caffemodel' % (MODEL_DIR, "lrcn_iter_110000") # Same caption
  #MODEL_FILE = '%s/%s.caffemodel' % ("/raid/anh/from_jeffdonahue_lrcn", "lrcn_caffenet_iter_110000")  # WORKING!
  #MODEL_FILE = '%s/%s.caffemodel' % ("/raid/anh/from_jeffdonahue_lrcn", "lrcn_caffenet_finetune_iter_100000")
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
  # generation_strategy = {'type': 'sample', 'temp': float("inf")}

  # if generation_strategy['type'] == 'beam':
  #   strategy_name = 'beam%d' % generation_strategy['beam_size']
  # elif generation_strategy['type'] == 'sample':
  #   strategy_name = 'sample%f' % generation_strategy['temp']
  # else:
  #   raise Exception('Unknown generation strategy type: %s' % generation_strategy['type'])

  image_path = "/home/anh/src/caffe_lrcn/images/bellpeppers.jpg"
  # image_path = "/home/anh/src/caffe_lrcn/images/brambling.jpg"
  experimenter = CaptionExperiment(captioner, image_path=image_path)
  captioner.set_image_batch_size(min(100, MAX_IMAGES))
  experimenter.generation_experiment(generation_strategy)

if __name__ == "__main__":
  main()
