from __future__ import division, print_function
import sys
from collections import defaultdict, Counter
import os
import numpy as np
import argparse
import random
import math
import six
import unicodedata
if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl

from utils import edist, edist_alt

def readPreds(fh):
    for line in fh:
        line = line.strip()
        if not line:
            continue

        if line.startswith("SRC:"):
            src = "".join(line[len("SRC:"):].split())
            src = src.replace("_", " ")
            src = src.replace("TRG LANG", "TRG_LANG")
        elif line.startswith("TRG:"):
            trg = "".join(line[len("TRG:"):].split())
            trg = trg.replace("_", " ")
        elif line.startswith("PRD:"):
            prd = "".join(line[len("PRD:"):].split())
            prd = prd.replace("_", " ")
            yield src, trg, prd
            src, trg, prd = (None, None, None)

def majoritySelector(preds):
    cts = Counter(preds)
    maj = cts.most_common(1)[0][0]
    for ind, pi in enumerate(preds):
        if pi == maj:
            return ind

def selector(sources):
    scores = [edistScore(source) for source in sources]
    #for si, sc in zip(sources, scores):
    #    print(si, sc)
    return np.argmin(scores)

def edistScore(source):
    origin, exemplar = source.split(":")
    exemplarLemma = exemplar.split(">")[0]
    score = edist(origin, exemplarLemma)
    return score

def caseVaries(insts):
    if len(insts) == 0:
        return False

    if np.all([correct is True for (src, targ, pred, correct) in insts]):
        return False
    if np.all([correct is False for (src, targ, pred, correct) in insts]):
        return False

    return True

def writeOfficial(paired, dev, ofh):
    adjScore = 0
    varScore = 0
    varCases = 0
    total = len(paired)
    totalCovered = 0
    for ind, insts in sorted(paired.items(), key=lambda xx: xx[0]):
        lemma, dTarg, feats = dev[ind]

        if insts:
            totalCovered += 1
            #select = selector([src for (src, targ, pred, correct) in insts])
            select = majoritySelector([pred for (src, targ, pred, correct) in insts])
            useInst = insts[select]

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
    if varCases == 0:
        selPerf = 1
    else:
        selPerf = varScore / varCases
    print("Selector performance: %d/%d %.2g" % (varScore, varCases, selPerf))
    print("Score within coverage: %d/%d %.2g" % (adjScore, totalCovered, adjScore / totalCovered))

def allowMatch(f1, f2):
    if f1 == f2:
        return True

    nbsp = unicodedata.lookup("NO-BREAK SPACE")

    for (c1, c2) in zip(f1, f2):
        if c1 != c2 and c1 != nbsp and c2 == nbsp:
            return False

    return True

if __name__ == "__main__":
    dev = sys.argv[1]
    prd = sys.argv[2]

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
                while not allowMatch(targ, nextForm) or len(paired[dInd]) == 5:
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
    totalInsts = 0
    correctInsts = 0
    for devItem, insts in paired.items():
        if len(insts) > 0:# == 5:
            coverage += 1
            totalInsts += len(insts)
            correctInsts += np.sum([int(correct) for (src, targ, pred, correct) in insts])
        else:
            #print(insts)
            #assert(len(insts) == 0)
            totalInsts += 5
            continue #not applicable to scoring rules

        if np.all([correct is True for (src, targ, pred, correct) in insts]):
            iScore += 1
        elif not np.all([correct is False for (src, targ, pred, correct) in insts]):
            variableItems += 1

    print("Coverage: %d/%d = %.2g" % (coverage, total, coverage/total))
    print("Variable items: %d/%d = %.2g" % (variableItems, total, variableItems/total))
    print("Score: %d/%d = %.2g" % (iScore, total, iScore/total))
    print("Score by instance: %d/%d = %.2g" % (correctInsts, totalInsts, correctInsts/totalInsts))

    ofName = os.path.dirname(prd) + "/predictions-std.txt"
    with open(ofName, "w") as ofh:
        writeOfficial(paired, rawData, ofh)
