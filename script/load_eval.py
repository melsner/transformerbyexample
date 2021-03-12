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

from s2sFlags import *

if __name__ == "__main__":
    #load the data
    print("Script start")
    args = get_arguments()
    print("Arguments parsed")
    run = args.run

    print("Parsing flags")
    variant = 0
    modelVariant = 0
    workdir = "inflect_%s/model%d" % (run, variant)
    while os.path.exists(workdir):
        if os.path.exists(workdir + "/checkpoints"):
            modelVariant = variant

        variant += 1
        workdir = "inflect_%s/model%d" % (run, variant)
    flags = S2SFlags(args, workdir)
    #flags.train = "inflect_%s/train.txt" % run
    flags.train = None
    flags.dev = "inflect_%s/dev.txt" % run
    if args.devset:
        flags.dev = args.devset
        workdir = "inflect_%s/%s" % (run, os.path.dirname(args.devset) + "-" + os.path.basename(args.devset))
        flags.work_dir = workdir

    #look for a checkpoint
    if args.load_other:
        print("Loading from pretrained checkpoint", args.load_other)
        flags.checkpoint_to_restore = os.path.abspath(args.load_other)

    elif variant > 0:
        print("Model dir exists, looking for checkpoint")
        cpdir = os.path.abspath("inflect_%s/model%d/checkpoints/" % (run, modelVariant))
        cpt = None
        for fi in os.listdir(cpdir):
            if fi.endswith(".index"):
                cpt = cpdir + "/" + fi
        assert(cpt is not None)
        print("Checkpoint", cpt)
        cpt.replace(".index", "")
        flags.checkpoint_to_restore = cpt

    print("Starting run")
    #import pdb
    #pdb.set_trace()
    seq2seq_runner.run(flags)
