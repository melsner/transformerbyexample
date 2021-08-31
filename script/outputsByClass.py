import sys
import os
import re
import csv
import numpy as np

from collections import defaultdict, Counter
from utils import readPreds, getEditClass

def statsByOutput(data):
    outToStats = defaultdict(Counter)
    classToOut = defaultdict(Counter)
    classEx = {}
    classPrs = defaultdict(list)

    for (targ, pred, correct, key, prob) in data:
        rule, nWords, ex = key.strip().split("\t")
        nWords = int(nWords)
        classToOut[rule][pred] += 1
        classEx[rule] = ex
        classPrs[rule].append(prob)
        outToStats[pred][(rule, nWords, ex)] += 1

    # for cls, sub in classToOut.items():
    #     print("Edit class:", cls, classEx[cls])
    #     print("Outputs:", sub.most_common())

    #     print()

    predToTotal = {}
    for pred, stats in outToStats.items():
        total = 0
        for (rule, nWords, ex) in stats:
            total += nWords
        predToTotal[pred] = total

    return outToStats, predToTotal, classPrs


def collate(data):
    outToStats, predToTotal, classPrs = statsByOutput(data)

    for pred, total in sorted(predToTotal.items(), key=lambda xx: xx[1], reverse=True):
        print("Output:", pred)
        stats = outToStats[pred]
        print("Supported by:", len(stats), "rules, total words:", total)
        for (rule, nWords, ex), ct in stats.items():
            prs = classPrs[rule]
            lowPr = min(prs)
            highPr = max(prs)
            medPr = np.median(prs)
            print("\t (%d samples):" % ct, rule, ex, ":\t", nWords, ":\t %.2f %.2f %.2f" % (lowPr, medPr, highPr))
        print()


if __name__ == "__main__":
    preds = open(sys.argv[1])
    classPreds = open(sys.argv[2])
    keys = open(sys.argv[3])

    prs = []
    for (src, targ, prob, correct) in readPreds(classPreds):
        prs.append(prob)

    bySource = defaultdict(list)
    for (src, targ, pred, correct), key, prob in zip(readPreds(preds), keys, prs):
        srcLemma = src.split(":")[0]
        bySource[srcLemma].append((targ, pred, correct, key, prob))

    for src, data in bySource.items():
        print("--------------")
        print(src)
        print()
        collate(data)
        print()
