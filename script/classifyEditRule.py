from __future__ import division, print_function
import sys
from collections import defaultdict, Counter
import os
import re
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

import tensorflow as tf
import tensorflow.keras as tkeras

import scipy.spatial.distance

from byexample import Data, getEditClass
from classifyEditUtil import *

def runModel(flags):
    tcls, data, devData = buildModel(flags)
    tcls.train(data, devData)

    findSimilarWords(tcls, devData)

def findSimilarWords(tcls, data):
    lemma, cell, rule = data.instances[0]
    uniqueWords = list(set([xx[0] for xx in data.instances]))
    #use dummy cell and rule, we won't need these
    insts = [(lemma, cell, rule) for lemma in uniqueWords]
    uniqueDat = ClassificationData(insts, data.lemmaIndex, data.cellIndex, data.ruleIndex,
                                   data.rules, data.maxLen, data.batchSize, truncate=False)
    reprs = tcls.wordReprs(uniqueDat)
    print("reprs shape", reprs.shape)
    dmat = scipy.spatial.distance.pdist(reprs, metric="cosine")
    dmat = scipy.spatial.distance.squareform(dmat)

    for ii in range(len(data.instances)):
        lemma, cell, rule = data.instances[ii]
        bestX = np.argsort(dmat[ii])[:10]
        bestXWords = [data.instances[xx][0] for xx in bestX]
        bestXDists = [dmat[ii, xx] for xx in bestX]
        print(lemma)
        for bi, bd in zip(bestXWords, bestXDists):
            print("\t", bi, bd)

def buildModel(flags):
    hparams, flags = seq2seq_runner.handle_preparation_flags(flags)

    clsData, dev = readData(hparams, flags)

    input_vocab_size = clsData.lemmaIndex.nChars

    encoder = model_lib.Encoder(
            hparams.num_layers, hparams.d_model, hparams.num_heads,
            hparams.dff, input_vocab_size, hparams.dropout_rate)

    print("Built an encoder")

    cellDims = [ ]
    for cell in clsData.cells:
        cellDims.append((cell, clsData.nOutcomes[cell]))

    model = MultiOutModel(hparams, flags, encoder, cellDims)

    print("Built a model")

    return model, clsData, dev

def writeEditData(data, ofh, dev=False):
    if dev:
        instances = data.devSet
    else:
        instances = data.instances

    for (lemma, form, feats, lang, fam) in instances:
        cell = "%s:%s" % (lang, ";".join(sorted(feats)))
        er = getEditClass(lemma, form)
        er = "".join(er)
        ofh.write("%s\t%s\t%s\n" % (lemma, cell, er))

if __name__ == "__main__":
    print("Script start")
    args = get_arguments()
    print("Arguments parsed")
    run = args.run
    dfile = args.data

    if os.path.isdir(dfile):
        allData = []
        for root, dirs, files in os.walk(dfile):
            for fi in files:
                lang = fi
                valid = False
                family = None

                if lang.endswith(".dev") or lang.endswith(".trn"):
                    #set lang fam for 2020
                    family = os.path.basename(root)

                for code in ["-train-low", "-train-high", ".trn"]:
                    if code in lang:
                        lang = lang.replace(code, "")
                        valid = True

                if not valid:
                    continue
                    
                print("Reading", fi, "for", lang, "in", family)

                rawData = np.loadtxt(root + "/" + fi, dtype=str, delimiter="\t")
                allData.append((rawData, lang, family))
                
        data = Data(allData, lang=None, family=None, nExemplars=args.n_exemplars, useEditClass=args.edit_class)
    else:
        lang = os.path.basename(dfile)
        family = None
        if lang.endswith(".dev") or lang.endswith(".trn"):
            #set lang fam for 2020
            family = os.path.basename(os.path.dirname(dfile))

        for code in ["-dev", "-test", "-train-low", "-train-high", ".trn", ".dev"]:
            lang = lang.replace(code, "")

        print("Running single dataset for", lang, "in", family)
        rawData = np.loadtxt(dfile, dtype=str, delimiter="\t")
        data = Data(rawData, lang=lang, family=family, nExemplars=args.n_exemplars, useEditClass=args.edit_class)

    data.splitDev(instances=1000)

    os.makedirs(run, exist_ok=True)
    with open("%s/train.txt" % run, "w") as ofh:
        writeEditData(data, ofh)

    with open("%s/dev.txt" % run, "w") as ofh:
        writeEditData(data, ofh, dev=True)

    variant = 0
    scratchDir = args.run + "/model%d" % variant
    while os.path.exists(scratchDir):
        variant += 1
        scratchDir = os.path.basename(args.run) + "/model%d" % variant

    flags = S2SFlags(args, scratchDir)
    flags.train = args.run + "/train.txt"
    flags.dev = args.run + "/dev.txt"

    print("Starting run")
    runModel(flags)
