from __future__ import division, print_function
import sys
from collections import defaultdict
import re
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

import tensorflow as tf
import tensorflow.keras as tkeras

from classifyVariability import *

def readPreds(fh):
    for line in fh:
        line = line.strip()
        if not line:
            continue

        if line.startswith("SRC:"):
            src = "".join(line[len("SRC:"):].split())
            src = src.replace("_", " ")
            src = src.replace("TRG LANG ", "TRG_LANG_")
        elif line.startswith("TRG:"):
            trg = "".join(line[len("TRG:"):].split())
            trg = trg.replace("_", " ")
        elif line.startswith("PRD:"):
            prd = "".join(line[len("PRD:"):].split())
            prd = prd.replace("_", " ")
            yield src, trg, prd
            src, trg, prd = (None, None, None)

def prepSource(src):
    langPatt = "TRG_LANG_(.*)"
    lang = re.search(langPatt, src).group(0)
    src = re.sub(langPatt, "", src)
    #print(src, lang)
    prepped = src.replace(" ", "_")
    prepped = list(src)
    prepped = prepped + [lang,]
    prepped = " ".join(prepped)
    return prepped

def selector(sources, model):
    #print(sources)
    prepped = [prepSource(src) for src in sources]
    #print(prepped)
    reprs = model.prepare_for_forced_validation(prepped, model.src_language_index)
    enc_padding_mask = model_lib.create_padding_mask(reprs)
    scores = model.transformer.call(reprs, False, enc_padding_mask)

    return np.argmax(scores), scores

def edistScore(source):
    origin, exemplar = source.split(":")
    exemplarLemma = exemplar.split(">")[0]
    score = edist(origin, exemplarLemma)
    return score

CACHE = {}
def edist(s1, s2):
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    if (s1, s2) in CACHE:
        return CACHE[(s1, s2)]

    if s1[-1] == s2[-1]:
        cost = 0
    else:
        cost = 2

    op1 = edist(s1[:-1], s2) + 1
    op2 = edist(s1, s2[:-1]) + 1
    op3 = edist(s1[:-1], s2[:-1]) + cost
    mincost = min(op1, min(op2, op3))
    CACHE[(s1, s2)] = mincost
    return mincost

def caseVaries(insts):
    if len(insts) == 0:
        return False

    if np.all([correct is True for (src, targ, pred, correct) in insts]):
        return False
    if np.all([correct is False for (src, targ, pred, correct) in insts]):
        return False

    return True

def writeOfficial(paired, dev, ofh, model):
    adjScore = 0
    varScore = 0
    varCases = 0
    total = len(paired)
    totalCovered = 0
    for ind, insts in sorted(paired.items(), key=lambda xx: xx[0]):
        lemma, dTarg, feats = dev[ind]

        if insts:
            totalCovered += 1
            select, scores = selector([src for (src, targ, pred, correct) in insts], model)
            useInst = insts[select]

            if caseVaries(insts):
                for ind, (inst, sc) in enumerate(zip(insts, scores)):
                    src, targ, pred, correct = inst
                    if ind == select:
                        star = "*"
                    else:
                        star = ""
                    print(src, "\t", targ, "\t", pred, "\t\t", sc.numpy(), correct, star)
                print()

            src, targ, pred, correct = useInst

            ofh.write("%s\t%s\t%s\n" % (lemma, pred, feats))

            adjScore += int(useInst[-1])
            if caseVaries(insts):
                varCases += 1
                if useInst[-1]:
                    varScore += 1
        else:
            ofh.write("%s\t%s\t%s\n" % (lemma, lemma, feats))

    print("Adjusted score: %d/%d %.2g" % (adjScore, total, adjScore / total))
    print("Selector peformance: %d/%d %.2g" % (varScore, varCases, varScore / varCases))
    print("Score within coverage: %d/%d %.2g" % (adjScore, totalCovered, adjScore / totalCovered))

if __name__ == "__main__":
    print("Script start")
    args = get_arguments()
    print("Arguments parsed")
    prd = args.run
    dev = args.data

    assert(args.load_other is not None)
    if not args.load_other.endswith(".index"):
        print("Searching for latest checkpoint")
        best = 0
        bestC = None
        lst = os.listdir(args.load_other)

        for cpt in lst:
            if cpt.endswith(".index"):
                cptN = int(cpt.replace(".index", "").split("-")[1])
                if cptN > best:
                    best = cptN
                    bestC = cpt

        assert(bestC is not None)
        args.load_other += "/" + cpt
        print("Using", args.load_other)

    variant = 0
    workdir = os.path.dirname(args.load_other) + "/../../classify-variable%d" % variant
    while os.path.exists(workdir):
        variant += 1
        workdir = os.path.dirname(args.load_other) + "/../../classify-variable%d" % variant

    flags = S2SFlags(args, workdir)
    flags.train = os.path.dirname(args.load_other) + "/../../dev.txt"
    flags.dev = os.path.dirname(args.load_other) + "/../../dev.txt"
    flags.checkpoint_to_restore = os.path.abspath(args.load_other)
    tcls = buildModel(flags)
    tcls.train(fake=True)
    print(type(tcls.transformer))
    print(tcls.transformer.layers)
    tcls.transformer.load_weights(args.placeholder_load)
    print("Loaded model")

    rawData = np.loadtxt(dev, dtype=str, delimiter="\t")
    dIter = iter(enumerate(rawData))
    paired = {}

    for ii in range(rawData.shape[0]):
        paired[ii] = []

    with open(prd) as prdFh:
        dInd, (nextLemma, nextForm, nextFeats) = next(dIter)
        for src, targ, pred in readPreds(prdFh):
            #print("Looking for", targ, "vs", nextForm)
            try:
                while targ != nextForm or len(paired[dInd]) == 5:
                    dInd, (nextLemma, nextForm, nextFeats) = next(dIter)
                    #print("\tcycle", nextForm)
            except StopIteration:
                print("Last item not detected; probably fine, check dev file")

            correct = (targ == pred)

            paired[dInd].append((src, targ, pred, correct))


    coverage = 0
    variableItems = 0
    iScore = 0
    total = len(paired)
    for devItem, insts in paired.items():
        if len(insts) == 5:
            coverage += 1
        else:
            assert(len(insts) == 0)
            continue #not applicable to scoring rules

        if np.all([correct is True for (src, targ, pred, correct) in insts]):
            iScore += 1
        elif not np.all([correct is False for (src, targ, pred, correct) in insts]):
            variableItems += 1

    print("Coverage: %d/%d = %.2g" % (coverage, total, coverage/total))
    print("Variable items: %d/%d = %.2g" % (variableItems, total, variableItems/total))
    print("Score: %d/%d = %.2g" % (iScore, total, iScore/total))

    ofName = os.path.dirname(prd) + "/predictions-std.txt"
    with open(ofName, "w") as ofh:
        writeOfficial(paired, rawData, ofh, tcls)
