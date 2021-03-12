from __future__ import division, print_function
from sys import argv, exit, stderr, stdout
import os
from sklearn import manifold
from sklearn.cluster import KMeans
from scipy.special import softmax
import numpy as np
import argparse
from segment_functions import *
from initialize import *
from matching import *
import multiprocessing as mp
import random
import torch
import math
import six
if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl
from Seq2seq import seq2seq_runner

class Inflector:

    def __init__(self, args, MODEL_DIR, extrinsic=False):

        self.work_dir = MODEL_DIR
        self.supervision = args.supervision
        self.train = None
        self.dev = None
        self.test = None
        self.dev_acc = 0.0
        self.preTrained_context_embeddings = None
        self.checkpoint_to_restore = None
        self.data_format = args.s2s_data_format
        self.model = args.s2s_model
        self.max_num_epochs = args.s2s_max_num_epochs
        self.patience = args.s2s_patience
        self.batch_size = args.s2s_batch_size
        self.val_batch_size = args.s2s_val_batch_size
        self.optimizer = args.s2s_optimizer
        self.epsilon = args.s2s_epsilon
        self.num_layers = args.s2s_num_layers
        self.d_model = args.s2s_d_model
        self.num_heads = args.s2s_num_heads
        self.dff = args.s2s_dff
        self.dropout_rate = args.s2s_dropout_rate
        self.beta_1 = args.s2s_beta_1
        self.beta_2 = args.s2s_beta_2
        self.warmup_steps = args.s2s_warmup_steps
        self.trained_model = None

        if extrinsic:  # We might have considered reducing the size of the model during EM for efficiency, but the final extrinsic model should be trained with full hyperparameters.
            self.max_num_epochs = 100
            self.patience = 40
            self.num_layers = 4
            self.d_model = 128
            self.dff = 512

    def load_preTrained_context_embeddings(self, preTrained_context_embeddings):
        self.preTrained_context_embeddings = preTrained_context_embeddings
        
    def prepare_s2s(self, train, dev=None, test=None):

        # Prepare to start training
        self.start_epoch = 0

        # TODO: handle pretrained context embeddings..

        if 'lemma' not in self.supervision:
            raise Exception('Need to strip base inds before passing to seq2seq.. I have not yet coded this')

        fn = os.path.join(os.path.dirname(self.work_dir), '{}_train.txt'.format(os.path.basename(self.work_dir)))
        with open(fn, 'w') as out_file:
            write_out_data_by_step(train, out_file)
        self.train = fn

        if dev:
            fn = os.path.join(os.path.dirname(self.work_dir), '{}_dev.txt'.format(os.path.basename(self.work_dir)))
            with open(fn, 'w') as out_file:
                write_out_data_by_step(dev, out_file)
            self.dev = fn

        if test:
            fn = os.path.join(os.path.dirname(self.work_dir), '{}_test.txt'.format(os.path.basename(self.work_dir)))
            with open(fn, 'w') as out_file:
                write_out_data_by_step(test, out_file)
            self.test = fn

    def continue_s2s(self, train, dev=None, test=None):

        # Pickup where we left off training
        raise NotImplementedError

    def train_validate_s2s(self):

        trained_model = seq2seq_runner.run(self, mode='ANA')
        self.dev_acc = trained_model.dev_acc
        self.trained_model = trained_model

        return trained_model.base_wf_tags_2_loss

    def train_validate_s2s_extrinsic(self):

        trained_model = seq2seq_runner.run(self)
        self.dev_acc = trained_model.dev_acc
        self.test_acc = trained_model.test_acc
        self.trained_model = trained_model

        return trained_model

def write_out_data_by_step(dataset, out_file):
    for (lem, wf, tup) in dataset:

        cluster, IC, context_vector_idx = tup
        printline = '{}\t{}\t{}'.format(lem, wf, cluster)
        if IC != None:
            printline += '\t{}'.format(IC)
        if context_vector_idx != None:
            printline += '\t{}'.format(context_vector_idx)

        out_file.write('{}\n'.format(printline))






