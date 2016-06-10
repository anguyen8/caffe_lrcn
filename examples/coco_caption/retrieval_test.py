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
  def __init__(self, captioner, dataset, dataset_cache_dir, cache_dir):
    self.captioner = captioner
    self.dataset_cache_dir = dataset_cache_dir
    self.cache_dir = cache_dir
    for d in [dataset_cache_dir, cache_dir]:
      if not os.path.exists(d): os.makedirs(d)
    self.dataset = dataset
    self.images = dataset.keys()
    self.init_caption_list(dataset)
    self.caption_scores = [None] * len(self.images)

    print 'Initialized caption experiment: %d images, %d captions' % \
        (len(self.images), len(self.captions))

    print "+++", self.images

  def init_caption_list(self, dataset):
    self.captions = []
    for image, captions in dataset.iteritems():
      for caption, _ in captions:
        self.captions.append({'source_image': image, 'caption': caption})
    # Sort by length for performance.
    self.captions.sort(key=lambda c: len(c['caption']))

  def compute_descriptors(self):
    descriptor_filename = '%s/descriptors.npz' % self.dataset_cache_dir
    if os.path.exists(descriptor_filename):
      self.descriptors = np.load(descriptor_filename)['descriptors']
    else:
      self.descriptors = self.captioner.compute_descriptors(self.images)
      np.savez_compressed(descriptor_filename, descriptors=self.descriptors)


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
    all_captions = [None] * num_images
    for image_index in xrange(0, num_images, batch_size):
      batch_end_index = min(image_index + batch_size, num_images)
      sys.stdout.write("\rGenerating captions for image %d/%d" %
                       (image_index, num_images))
      sys.stdout.flush()
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
          all_captions[batch_index] = output

    # Collect model/reference captions, formatting the model's captions and
    # each set of reference captions as a list of len(self.images) strings.
    exp_dir = '%s/generation' % self.cache_dir
    if not os.path.exists(exp_dir):
      os.makedirs(exp_dir)
    # For each image, write out the highest probability caption.
    model_captions = [''] * len(self.images)
    # reference_captions = [([''] * len(self.images)) for _ in xrange(num_reference_files)]
    for image_index, image in enumerate(self.images):
      print "+", image

      caption = self.captioner.sentence(all_captions[image_index])

      print ">>>> : [", caption, "]"


def main():
  MAX_IMAGES = 1  # -1 to use all images
  TAG = 'coco_2layer_factored'
  if MAX_IMAGES >= 0:
    TAG += '_%dimages' % MAX_IMAGES
  eval_on_test = False
  if eval_on_test:
    ITER = 100000
    MODEL_FILENAME = 'lrcn_finetune_trainval_stepsize40k_iter_%d' % ITER
    DATASET_NAME = 'test'
  else:  # eval on val
    #ITER = 100000
    ITER = 110000
    #MODEL_FILENAME = 'lrcn_vgg_iter_%d' % ITER
    #MODEL_FILENAME = 'lrcn_caffenet_finetune_iter_%d' % ITER
    MODEL_FILENAME = 'lrcn_caffenet_iter_%d' % ITER
    DATASET_NAME = 'val'

  TAG += '_%s' % DATASET_NAME
  #MODEL_DIR = './examples/coco_caption'
  MODEL_DIR = "/raid/anh/from_jeffdonahue_lrcn"
  MODEL_FILE = '%s/%s.caffemodel' % (MODEL_DIR, MODEL_FILENAME)
  #MODEL_FILE = '%s/%s.caffemodel' % (MODEL_DIR, "lrcn_iter_110000") # Same caption
  #MODEL_FILE = '%s/%s.caffemodel' % ("/raid/anh/from_jeffdonahue_lrcn", "lrcn_caffenet_iter_110000")  # WORKING!
  #MODEL_FILE = '%s/%s.caffemodel' % ("/raid/anh/from_jeffdonahue_lrcn", "lrcn_caffenet_finetune_iter_100000")
  IMAGE_NET_FILE = './models/bvlc_reference_caffenet/deploy.prototxt'
  LSTM_NET_FILE = './examples/coco_caption/lrcn_word_to_preds.deploy.prototxt'
  NET_TAG = '%s_%s' % (TAG, MODEL_FILENAME)
  DATASET_SUBDIR = '%s/%s_ims' % (DATASET_NAME,
      str(MAX_IMAGES) if MAX_IMAGES >= 0 else 'all')
  DATASET_CACHE_DIR = './retrieval_cache/%s/%s' % (DATASET_SUBDIR, MODEL_FILENAME)
  VOCAB_FILE = './examples/coco_caption/h5_data/buffer_100/vocabulary.txt'
  DEVICE_ID = 0
  with open(VOCAB_FILE, 'r') as vocab_file:
    vocab = [line.strip() for line in vocab_file.readlines()]
  
  coco = COCO(COCO_ANNO_PATH % DATASET_NAME)
  image_root = COCO_IMAGE_PATTERN % DATASET_NAME
  sg = CocoSequenceGenerator(coco, BUFFER_SIZE, image_root, vocab=vocab,
                             align=False, shuffle=False)
  dataset = {}
  dataset["/home/anh/src/caffe_lrcn/images/brambling.jpg"] = [("", "")]

  # for image_path, sentence in sg.image_sentence_pairs:
  #   if image_path not in dataset:
  #     dataset[image_path] = []
  #   dataset[image_path].append((sg.line_to_stream(sentence), sentence))


  # print 'Original dataset contains %d images' % len(dataset.keys())
  # if 0 <= MAX_IMAGES < len(dataset.keys()):
  #   all_keys = dataset.keys()
  #   perm = np.random.permutation(len(all_keys))[:MAX_IMAGES]
  #   chosen_keys = set([all_keys[p] for p in perm])
  #   for key in all_keys:
  #     if key not in chosen_keys:
  #       del dataset[key]
  #   print 'Reduced dataset to %d images' % len(dataset.keys())
  if MAX_IMAGES < 0: MAX_IMAGES = len(dataset.keys())
  captioner = Captioner(MODEL_FILE, IMAGE_NET_FILE, LSTM_NET_FILE, VOCAB_FILE,
                        device_id=DEVICE_ID)
  beam_size = 1
  generation_strategy = {'type': 'beam', 'beam_size': beam_size}
  if generation_strategy['type'] == 'beam':
    strategy_name = 'beam%d' % generation_strategy['beam_size']
  elif generation_strategy['type'] == 'sample':
    strategy_name = 'sample%f' % generation_strategy['temp']
  else:
    raise Exception('Unknown generation strategy type: %s' % generation_strategy['type'])
  CACHE_DIR = '%s/%s' % (DATASET_CACHE_DIR, strategy_name)
  experimenter = CaptionExperiment(captioner, dataset, DATASET_CACHE_DIR, CACHE_DIR, sg)
  captioner.set_image_batch_size(min(100, MAX_IMAGES))
  experimenter.generation_experiment(generation_strategy)
  # captioner.set_caption_batch_size(min(MAX_IMAGES * 5, 1000))
  # experimenter.retrieval_experiment()

if __name__ == "__main__":
  main()
