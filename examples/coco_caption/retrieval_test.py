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

from coco_to_hdf5_data import *
from captioner import Captioner

COCO_EVAL_PATH = './data/coco/coco-caption-eval'
sys.path.append(COCO_EVAL_PATH)
from pycocoevalcap.eval import COCOEvalCap

class CaptionExperiment():
  # captioner is an initialized Captioner (captioner.py)
  # dataset is a dict: image path -> [caption1, caption2, ...]
  def __init__(self, captioner, dataset):
    self.captioner = captioner
    
    self.images = dataset.keys()
    self.init_caption_list(dataset)

    print 'Initialized caption experiment: %d images, %d captions' % \
        (len(self.images), len(self.captions))

  def init_caption_list(self, dataset):
    self.captions = []
    for image, captions in dataset.iteritems():
      for caption, _ in captions:
        self.captions.append({'source_image': image, 'caption': caption})
    # Sort by length for performance.
    self.captions.sort(key=lambda c: len(c['caption']))

  def compute_descriptors(self):
    self.descriptors = self.captioner.compute_descriptors(self.images)

  def generation_experiment(self, strategy, max_batch_size=1000):
    # Compute image descriptors.
    print 'Computing image descriptors'
    self.compute_descriptors()

    do_batches = (strategy['type'] == 'beam' and strategy['beam_size'] == 1) or \
        (strategy['type'] == 'sample' and
         ('temp' not in strategy or strategy['temp'] in (1, float('inf'))) and
         ('num' not in strategy or strategy['num'] == 1))

    num_images = len(self.images)
    batch_size = min(max_batch_size, num_images) if do_batches else 1

    # Generate captions for all images.
    caption = None
    for image_index in xrange(0, num_images, batch_size):
      batch_end_index = min(image_index + batch_size, num_images)
      sys.stdout.write("\rGenerating captions for image %d/%d" %
                       (image_index, num_images))
      sys.stdout.flush()

      # print "==== do_batches", do_batches # TRue

      if do_batches:
        if strategy['type'] == 'beam' or \
            ('temp' in strategy and strategy['temp'] == float('inf')):
          temp = float('inf')
        else:
          temp = strategy['temp'] if 'temp' in strategy else 1
        output_captions, output_probs = self.captioner.sample_captions(
            self.descriptors[image_index:batch_end_index], temp=temp)

        #print ">>> output_captions", output_captions

        for batch_index, output in zip(range(image_index, batch_end_index),
                                       output_captions):
          caption = output

    # Collect model/reference captions, formatting the model's captions and
    # each set of reference captions as a list of len(self.images) strings.
    # For each image, write out the highest probability caption.
    model_captions = [''] * len(self.images)

    # reference_captions = [([''] * len(self.images)) for _ in xrange(num_reference_files)]
    for image_index, image in enumerate(self.images):
      print "+", image

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
  
  dataset = {}
  dataset["/home/anh/src/caffe_lrcn/images/brambling.jpg"] = [("", "")]

  if MAX_IMAGES < 0: MAX_IMAGES = len(dataset.keys())
  captioner = Captioner(MODEL_FILE, IMAGE_NET_FILE, LSTM_NET_FILE, VOCAB_FILE,
                        device_id=DEVICE_ID)
  beam_size = 1
  # generation_strategy = {'type': 'beam', 'beam_size': beam_size}
  generation_strategy = {'type': 'sample', 'temp': "inf"}


  if generation_strategy['type'] == 'beam':
    strategy_name = 'beam%d' % generation_strategy['beam_size']
  elif generation_strategy['type'] == 'sample':
    strategy_name = 'sample%f' % generation_strategy['temp']
  else:
    raise Exception('Unknown generation strategy type: %s' % generation_strategy['type'])

  experimenter = CaptionExperiment(captioner, dataset)
  captioner.set_image_batch_size(min(100, MAX_IMAGES))
  experimenter.generation_experiment(generation_strategy)

if __name__ == "__main__":
  main()
