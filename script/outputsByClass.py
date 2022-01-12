import sys
import os
import re
import csv
import numpy as np

from collections import defaultdict, Counter
from utils import readPreds, getEditClass

def statsByOutput(data):
    #rule : { output : list of prs }
    ruleOutputs = defaultdict(lambda: defaultdict(list))
    ruleToInfo = {}
    seenExes = set()

    for (exemplar, targ, pred, correct, key, prob) in data:
        rule, nWords, wMax, ruleEx = key.strip().split("\t")
        nWords = int(nWords)
        wMax = float(wMax)
        ruleToInfo[rule] = (ruleEx, nWords, wMax)
        if exemplar in seenExes:
            print("Warning: saw exemplar twice", exemplar)
            continue

        seenExes.add(exemplar)
        ruleOutputs[rule][pred].append(prob)

    #pred : [ (rule, nWords, ex, prs) ]
    outToStats = defaultdict(list)
    predToTotal = Counter()
    for rule, sub in ruleOutputs.items():
        totalExes = sum([len(xx) for xx in sub.values()])
        ruleEx, nWords, wMax = ruleToInfo[rule]

        for pred, prs in sub.items():
            supportingExes = len(prs)
            proportion = supportingExes / totalExes
            supportingWords = nWords * proportion

            outToStats[pred].append( (rule, supportingWords, wMax, ruleEx, prs) )
            predToTotal[pred] += supportingWords

    return outToStats, predToTotal


def collate(data):
    outToStats, predToTotal = statsByOutput(data)

    for pred, total in sorted(predToTotal.items(), key=lambda xx: xx[1], reverse=True):
        print("Output:", pred)
        stats = outToStats[pred]
        print("Supported by:", len(stats), "rules, total words:", total)
        for (rule, nWords, wMax, ex, prs) in stats:
            lowPr = min(prs)
            highPr = max(prs)
            medPr = np.median(prs)
            print("\t (%d samples):" % len(prs), rule, ex, ":\t", nWords, ":\t %.2f %.2f %.2f" % (lowPr, medPr, highPr))
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
        srcLemma, exe = src.split(":")
        bySource[srcLemma].append((exe, targ, pred, correct, key, prob))

    for src, data in bySource.items():
        print("--------------")
        print(src)
        print()
        collate(data)
        print()
