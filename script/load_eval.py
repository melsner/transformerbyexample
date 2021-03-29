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

    os.makedirs(run, exist_ok=True)

    print("Parsing flags")
    variant = 0
    modelVariant = 0
    workdir = "%s/model%d" % (run, variant)
    while os.path.exists(workdir):
        if os.path.exists(workdir + "/checkpoints"):
            modelVariant = variant

        variant += 1
        workdir = "%s/model%d" % (run, variant)
    flags = S2SFlags(args, workdir)
    flags.train = None
    dev = "%s/dev.txt" % run
    if args.devset:
        dev = args.devset
        workdir = "%s/%s" % (run, os.path.basename(os.path.dirname(args.devset)) + "-" + os.path.basename(args.devset))
        print("Naming work dir", workdir)
        flags.work_dir = workdir

    #look for a checkpoint
    if args.load_other:
        if not args.load_other.endswith(".index"):
            print("Searching for latest checkpoint")
            best = 0
            bestC = None

            for cpt in os.listdir(args.load_other):
                if cpt.endswith(".index"):
                    cptN = int(cpt.replace(".index", "").split("-")[1])
                    if cptN > best:
                        best = cptN
                        bestC = cpt

            assert(bestC is not None)
            args.load_other += "/" + bestC
            print("Using", args.load_other)
        else:
            print("Loading from pretrained checkpoint", args.load_other)

        flags.checkpoint_to_restore = os.path.abspath(args.load_other)

    elif variant > 0:
        print("Model dir exists, looking for checkpoint")
        cpdir = os.path.abspath("%s/model%d/checkpoints/" % (run, variant - 1))
        cpt = None
        best = 0
        for fi in os.listdir(cpdir):
            if fi.endswith(".index"):
                cptN = int(fi.replace(".index", "").split("-")[1])
                if cptN > best:
                    cpt = cpdir + "/" + fi
                    best = cptN
        assert(cpt is not None)
        cpt.replace(".index", "")
        print("Checkpoint", cpt)
        flags.checkpoint_to_restore = cpt

    print("Starting run")

    #create a scratch dev file for loading purposes
    with open(dev) as fh:
        with open(run + "/cdev.txt", "w") as ofh:
            for ind in range(50):
                ofh.write(next(fh))

    flags.dev = run + "/cdev.txt"

    hparams, flags = seq2seq_runner.handle_preparation_flags(flags)

    # Prepare data.
    all_data = seq2seq_runner.prepare_data(flags, hparams)
    trg_language_index = all_data.trg_language_index
    trg_feature_index = all_data.trg_feature_index
    trg_max_len_seq = all_data.trg_max_len_seq
    trg_max_len_ft = all_data.trg_max_len_ft
    split_sizes = all_data.split_sizes

    # Get model.
    model = model_lib.Model(hparams, all_data, flags)

    print("Created model")

    resultsFile = workdir + "/predictions_dev.txt"

    done = False
    with open(dev) as fh:
        with open(resultsFile, "w") as ofh:
            bsize = hparams.val_batch_size
            while not done:
                batch = []
                targets = []
                for ind in range(bsize):
                    try:
                        line = next(fh)
                    except StopIteration:
                        done = True
                        continue

                    src, targ, feats = line.strip().split("\t")
                    src = src.replace(" ", "_")
                    src = " ".join(list(src))
                    feats = feats.split(";")
                    feats = ["TRG_%s" % fi for fi in feats]
                    src += " ".join(feats)

                    targ = targ.replace(" ", "_")
                    targ = " ".join(list(targ))

                    batch.append(src)
                    targets.append(targ)

                out = model.translate(batch)
                for (src, pred, targ) in zip(batch, out, targets):
                    if targ != pred:
                        ofh.write("*ERROR*\n")
                    ofh.write("SRC: {}\n".format(src))
                    ofh.write("TRG: {}\n".format(targ))
                    ofh.write("PRD: {}\n".format(pred))
                    ofh.write("\n")
                
    #import pdb
    #pdb.set_trace()
    #seq2seq_runner.run(flags)
