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
from shutil import copyfile

from Seq2seq import seq2seq_runner, dataloader
from Seq2seq import model as model_lib

from s2sFlags import *

import tensorflow as tf
import tensorflow.keras as tkeras

import scipy.spatial.distance

from byexample import Data, getEditClass
from classifyEditUtil import *
from utils import findLatestModel

def runModel(flags):
    tcls, data, devData = buildModel(flags)
    tcls.train(data, devData)

    findSimilarWords(tcls, devData)

def findSimilarWords(tcls, data, maxWords=None, showCell=None):
    lemma, cell, rule = data.instances[0]
    uniqueWords = list(set([xx[0] for xx in data.instances]))
    print("Finding similar words across", len(uniqueWords), "words")
    #use dummy cell and rule, we won't need these
    insts = [(lemma, cell, rule) for lemma in uniqueWords]
    if maxWords is not None:
        insts = insts[:maxWords]
    uniqueDat = ClassificationData(insts, data.lemmaIndex, data.cellIndex, data.ruleIndex,
                                   data.rules, data.maxLen, data.batchSize, truncate=False)
    reprs = tcls.wordReprs(uniqueDat)
    print("reprs shape", reprs.shape)
    dmat = scipy.spatial.distance.pdist(reprs, metric="cosine")
    dmat = scipy.spatial.distance.squareform(dmat)

    data.byCell = {}
    for (li, ci, ri) in data.instances:
        data.byCell[li, ci] = ri

    for ii in range(len(insts)):
        lemma, cell, rule = insts[ii]
        bestX = np.argsort(dmat[ii])[:10]
        bestXWords = [insts[xx][0] for xx in bestX]
        bestXDists = [dmat[ii, xx] for xx in bestX]
        if showCell:
            print("".join(lemma), getCell(data, lemma, showCell))
        else:
            print("".join(lemma))

        for bi, bd in zip(bestXWords, bestXDists):
            if showCell:
                print("\t", "".join(bi), getCell(data, bi, showCell), bd)
            else:
                print("\t", "".join(bi), bd)

def getCell(data, lemma, cell):
    if (lemma, cell) in data.byCell:
        return data.byCell[lemma, cell]
    return None

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

    if flags.checkpoint_to_restore:
        restoreCheckpoint(flags, model, clsData)

    return model, clsData, dev

def restoreCheckpoint(flags, model, clsData):
    #find the vocab map file and copy it
    checkpoint = flags.checkpoint_to_restore
    vmap = os.path.dirname(checkpoint) + "/../vocab_map.tsv"
    copyfile(vmap, flags.work_dir + "/vocab_map.tsv")

    #load the checkpoint
    cpName = checkpoint.replace("ckpt-", "manual_save").replace(".index", ".h5")
    for (_, (inp, trg)) in enumerate(clsData):
        model.train_step(inp, trg)
        break

    model.transformer.load_weights(cpName)

def writeEditData(data, ofh, dev=False):
    if dev:
        instances = data.devSet
    else:
        instances = data.instances

    for (lemma, form, feats, lang, fam) in instances:
        fam = fam.lower()
        cell = "%s:%s" % (lang, ";".join(sorted(feats)))
        er = getEditClass(lemma, form)
        er = "".join(er)
        feats = ";".join(["LANG_%s" % lang, "FAM_%s" % fam])
        ofh.write("%s\t%s\t%s\t%s\n" % (lemma, cell, feats, er))

def upsample(instances, rate=.3):
    langCounts = Counter()
    for (lemma, form, feats, lang, fam) in instances:
        langCounts[lang] += 1

    largest = langCounts.most_common(1)
    print("Most resourced language:", largest)
    largest = largest[0][1] #count of the item

    multipliers = {}
    for lang, count in langCounts.items():
        mult = np.round((rate * largest) / count)
        mult = np.clip(mult, 1, 4)
        mult = int(mult)
        print("Multiplier for language", lang, "is", mult)
        multipliers[lang] = mult
    
    res = []
    for (lemma, form, feats, lang, fam) in instances:
        for ii in range(multipliers[lang]):
            res.append((lemma, form, feats, lang, fam))

    return res

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

    nDev = min(2000, int(.25 * len(data.instances)))
    data.splitDev(instances=nDev)

    os.makedirs(run, exist_ok=True)
    trainExists = os.path.exists("%s/train.txt" % run)
    if not trainExists:
        with open("%s/train.txt" % run, "w") as ofh:
            if args.upsample_classifier:
                data.instances = upsample(data.instances)
                print("Upsampled to", len(data.instances))

            writeEditData(data, ofh)

        with open("%s/dev.txt" % run, "w") as ofh:
            writeEditData(data, ofh, dev=True)

    variant = 0
    scratchDir = args.run + "/model%d" % variant
    checkpoint = None
    while os.path.exists(scratchDir):
        checkpoint = findLatestModel(scratchDir + "/checkpoints")
        variant += 1
        scratchDir = os.path.basename(args.run) + "/model%d" % variant

    flags = S2SFlags(args, scratchDir)
    flags.train = args.run + "/train.txt"
    flags.dev = args.run + "/dev.txt"
    print("Restoring", checkpoint)
    flags.checkpoint_to_restore = checkpoint

    print("Starting run")
    runModel(flags)
