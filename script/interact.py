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
from inflector import Inflector
import multiprocessing as mp
import random
import torch
import math
import six
import time
from collections import defaultdict, Counter
if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl
from Seq2seq.train import Data, Arch, Trainer
from Seq2seq.util import get_logger

if __name__ == '__main__':
    trainer = Trainer(get_logger("interaction-record.log", log_level="debug"), None)
    trainer.load_model(argv[1])
    trainer.load_data(Data.ANA, [
        ("a", [
            ("aditus", "aditu", (9, None, None))
            ]),
