from __future__ import division, print_function
import sys
from collections import defaultdict
import os
import numpy as np
import argparse
import random
import math
import six
if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl

from Seq2seq import seq2seq_runner, dataloader
from Seq2seq import model as model_lib

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--run", type=str, help="Name of the run directory to create.")
    parser.add_argument("-d", "--data", type=str, help="Path to the training file.")
    parser.add_argument("--load_other", type=str, default=None, help="Load a previously trained model.")
    parser.add_argument("--devset", type=str, default=None, help="Dev data for load/eval.")
    parser.add_argument("--generate_file", action="store_true", help="Just write instances to file and exit.")
    parser.add_argument("--append_train", action="store_true", help="If loading a previous model, append its training set to the training set.")
    parser.add_argument("--junk", action="store_true", help="Create junk affixes.")
    parser.add_argument("--src_length", type=int, default=-1, help="Source sequence length override.")
    parser.add_argument("--targ_length", type=int, default=-1, help="Target sequence length override.")
    parser.add_argument("--allow_self_exemplar", action="store_true", help="Training items are allowed to have themselves as exemplars (but not dev items).")

    parser.add_argument('-p', '--POS', type=str, choices=['all', 'N', 'V', 'ADJ'], help='This constrains what parts-of-speeches we train and test on.', default='all')

    parser.add_argument("--placeholder_load", type=str)
    parser.add_argument("--n_exemplars", type=str, default="5")
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--edit_class", type=str, default=None)

    ################################################################################
    ## COMPONENT SEQ 2 SEQ MODEL PARAMETERS
    ################################################################################
    parser.add_argument('--s2s_max_dev_size', default=200, type=int, help='maximum number of instances to calculate dev loss on.')
    parser.add_argument('--s2s_data_format', default=dataloader.DataFormat.INFLECTION, type=dataloader.DataFormat,
        # choices=['INFLECTION', 'INFLECTION_IC', MT'],
        help='Data files should contain one instance per line. This flag defines the '
          'syntax of instances. The following options are supported: '
          'MT: 2 tab delimited fields. (1) space-delimited source sentence (2) '
          'space-delimited target sentence. '
          'INFLECTION: 3 tab deimited fields. (1) lemma as undelimited characters '
          '(2) target as undelimeted characters (3) morphosyntactic property set '
          'describing (2). This is the standard UniMorph format. '
          'INFLECTION_IC: 4 tab deimited fields. Like INFLECTION but with an extra field for inflection class.'
          ### ADD ADDITIONAL FORMATS FOR HANDLING CONTEXT EMBEDDINGS.
          )
    parser.add_argument('--s2s_model', default=model_lib.ModelFormat.TRANSFORMER, type=model_lib.ModelFormat,
    # choices=['TRANSFORMER'],
    help='Only one architecture is supported at present: '
      'TRANSFORMER: Soft, non-monotonic multi-head attention without recurrent '
      'cells. Input is assumed to be one-dimensional with features treated as '
      'additional sequence elements, as in the KANN_2016 set up.'
      )
    # Defining model hyperparameters.
    parser.add_argument('--s2s_max_num_epochs', default=40, type=int, help='Maximum number of epochs.')
    parser.add_argument('--s2s_patience', default=10, type=int,
    help='The number of epochs the accuracy on the '
    'dev set is allowed not to improve before training stops.')
    parser.add_argument('--s2s_batch_size', default=64, type=int, help='Size of batches fed to the model.')
    parser.add_argument('--s2s_val_batch_size', default=1000, type=int, help='Size of batches during validation.')
    parser.add_argument('--s2s_optimizer', default='adam', type=str, choices=['adam', 'adadelta'], help='Optimization algorithm.')
    parser.add_argument('--s2s_epsilon', default=0.000000001, type=float, help='Small constant for stability.')
    # Hyperparameters specific to transformer model.
    # Notation is kept consistent with this implementation:
    # https://www.tensorflow.org/beta/tutorials/text/transformer
    parser.add_argument('--s2s_num_layers', default=4, type=int, help='Number of encoder and decoder layers.')
    parser.add_argument('--s2s_d_model', default=128, type=int, help='Similar to num_units in the RNN models.')
    parser.add_argument('--s2s_num_heads', default=8, type=int, help='Number of attention heads.')
    parser.add_argument('--s2s_dff', default=512, type=int, help='Dimensions in the feed forward network in each individual encoder and decoder layer.')
    parser.add_argument('--s2s_dropout_rate', default=0.1, type=float, help='Dropout rate.')
    parser.add_argument('--s2s_beta_1', default=0.9, type=float, help='Used for learning rate annealing.')
    parser.add_argument('--s2s_beta_2', default=0.98, type=float, help='Used for learning rate annealing.')
    parser.add_argument('--s2s_warmup_steps', default=4000, type=int, help='Learning rate will increase until this number of steps have been reached, then the learning rate will start annealing.')

    ################################################################################
    ## CONTEXT MODELING MODEL PARAMETERS
    ################################################################################
    parser.add_argument('--exponent_vocab', default=True, type=str2bool, help='Models distribution of exponents in context.')
    parser.add_argument('--model_context', default=True, type=str2bool, help='If False, no distributional data will be leveraged.')

    args = parser.parse_args()

    if args.n_exemplars not in ["all", "dynamic"]:
        args.n_exemplars = int(args.n_exemplars)

    return args


class S2SFlags:

    def __init__(self, args, MODEL_DIR):
        self.max_num_epochs = 100
        self.patience = 40
        self.num_layers = 4
        self.d_model = 128
        self.dff = 512

        self.src_length = args.src_length
        self.targ_length = args.targ_length

        self.work_dir = MODEL_DIR
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


