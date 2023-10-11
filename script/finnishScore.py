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
import re
import csv
if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl

from utils import getEditClass
from classifyVariability import readPreds, readCases

def process(bySource):
    outcomes = defaultdict(Counter)
    outcomesByRule = defaultdict(Counter)
    byTarget = defaultdict(Counter)
    everything = Counter()

    for src, data in bySource.items():
        for (exe, targ, pred, correct, key) in data:
            srcRule = getEditClass(src, targ)
            exeRule, nWords, wMax, ruleEx = key.strip().split("\t")
            exeRule = eval(exeRule)

            targRule = getEditClass(src, targ)
            predRule = getEditClass(src, pred)

            outcome = (pred, predRule)

            outcomes[src, targ, targRule][outcome] += 1

            outcomesByRule[srcRule][predRule] += 1

            everything[predRule] += 1

            byTarget[srcLemma, pred][exeRule] += 1

    ruleCode = {}
    for rule in outcomesByRule:
        if rule not in ruleCode:
            ruleCode[rule] = len(ruleCode)

    totalEnt = []
    for src, result in outcomes.items():
        rn = ruleCode.get(src[-1], None)
        print(src, "rule", rn)
        ent = 0
        total = sum(result.values())
        for ri, ct in result.most_common():
            print("\t", ri, ct)
            pi = ct / total
            ent += pi * np.log2(pi)
        print(":", -ent)
        totalEnt.append(-ent)
        print()

    print("Average entropy:", sum(totalEnt) / len(totalEnt))

    print("Rule summary")

    for srcRule, result in outcomesByRule.items():
        rn = ruleCode[srcRule]
        vals = result.values()
        total = sum(vals)
        prs = [val / total for val in vals]
        entropy = -1 * sum([pi * np.log2(pi) for pi in prs])
        print(srcRule, "rule", rn, entropy)
        for ri, ct in result.most_common():
            print("\t", ri, ct)
        print()

    total = sum(everything.values())
    prs = [ct / total for ct in everything.values()]
    entropy = -1 * sum([pi * np.log2(pi) for pi in prs])
    print("Entropy overall:", entropy)

    # sameOutcome = Counter()
    # for (src, pred), exes in byTarget.items():
    #     for ex, ct in exes.items():
    #         for ex2, ct2 in exes.items():
    #             if ex == ex2:
    #                 continue

    #             key = (".".join(ex), ".".join(ex2))
    #             sameOutcome[key] += min(ct, ct2)

    # with open(of, "w") as ofh:
    #     ofh.write("class1,class2,same.outcome\n")

    #     for key, ct in sameOutcome.most_common():
    #         ofh.write(",".join([key[0], key[1], str(ct)]) + "\n")


if __name__ == "__main__":
    preds = open(sys.argv[1])
    keys = open(sys.argv[2])
    of = sys.argv[3]

    bySource = defaultdict(lambda: defaultdict(list))
    for (src, targ, pred, correct), key in zip(readPreds(preds), keys):
        srcLemma, exe = src.split(":")
        cell = re.search("TRG_CELL_(.*?)TRG", exe).groups(1)[0]
        #print(exe)
        #print(cell)
        #assert(0)
        bySource[cell][srcLemma].append((exe, targ, pred, correct, key))

    for ci in bySource:
        print("------------------- process ", ci, "---------------")
        process(bySource[ci])
