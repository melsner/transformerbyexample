# coding=utf-8
from __future__ import division, print_function
import sys
from collections import defaultdict, Counter
import os
import csv
import re
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

from utils import getEditClass
from classifyVariability import readPreds, readCases
from ruleFeatures import ruleDescription, locateChunks, chunkRule

def myLocateChunks(seq):
    stemInds = []
    for ind, subseq in enumerate(seq):
        if subseq[0] == "*":
            stemInds.append(ind)

    res = []
    for ind, subseq in enumerate(seq):
        if ind in stemInds:
            stemBefore = None
            stemAfter = None
        else:
            stemBefore = stemInds and ind > stemInds[0]
            stemAfter = stemInds and ind < stemInds[-1]

        res.append((subseq, ind, stemBefore, stemAfter))

    return res

def ruleType(rule, cv=False, addRm=False):
    vowels = "aeiouáéíóú"

    res = set()

    if rule == ("*",):
        res.add("RULE_SYNCRETIC")

    for chunk, chunkInd, stemBefore, stemAfter in myLocateChunks(chunkRule(rule)):
        #print(chunk, chunkInd, stemBefore, stemAfter)

        vocalic, consonantal = False, False
        for element in chunk:
            if len(element) > 1 and element[1] in vowels:
                vocalic = True
            elif len(element) > 1:
                consonantal = True

        phonStr = ""
        if cv and consonantal:
            phonStr += "C"
        if cv and vocalic:
            phonStr += "V"

        procStr = ""
        if addRm and "-" in chunk[0]:
            procStr += "REMOVE"
        elif addRm and "+" in chunk[0]:
            procStr += "ADD"

        if chunk == ["*"]:
            continue
        elif not stemBefore:
            res.add("RULE_PREF_%s%s" % (procStr, phonStr))
        elif not stemAfter:
            res.add("RULE_SUFF_%s%s" % (procStr, phonStr))
        elif stemBefore and stemAfter:
            res.add("RULE_STEM_%s%s" % (procStr, phonStr))
        else:
            #pathological: rule with no identifiable stem content
            res.add("RULE_SUPPLETIVE")

    res = sorted(list(res))

    return res

def digraphify(word):
    vowels = "aeiouáéíóú"
    #note for welsh: should handle y, w, ngh
    letters = list(word)
    for ind, li in enumerate(letters):
        if ind < len(letters) - 1 and letters[ind + 1] == "h" and letters[ind] not in vowels:
            letters[ind] += "h"
            letters[ind + 1] = ""

    return tuple([xx for xx in letters if xx != ""])

def scoreDir(fpath):
    row = {}
    print(fpath, os.path.abspath(fpath))
    mtch = re.search("/((?:mono)|(?:[a-z]+)_trans)/(?:mono|trans)_([0-9]+)_x([0-9])", os.path.abspath(fpath))
    print(mtch.groups())
    lang, tsize, run = mtch.groups()
    row["lang"] = lang
    row["train.size"] = tsize
    row["run"] = run

    stem = os.path.basename(fpath.rstrip("/"))
    predDir = fpath + "/" + stem + "-dev.txt-predictions_dev-predicted.txt/"
    fpath = predDir + "/predictions_dev.txt"
    print("Checking", fpath)

    plurTypes = Counter()
    typeExes = defaultdict(list)
    accByClass = defaultdict(list)

    accFeats = defaultdict(list)

    tscore = 0
    total = 0

    actualOutputs = Counter()

    rulesByType = defaultdict(lambda: defaultdict(list))

    specificRuleConf = Counter()

    with open(fpath) as fh:
        preds = list(readPreds(fh))

        for (src, targ, pred, correct) in preds:
            if correct:
                tscore += 1
            total += 1

            lemma = src[:src.index(":")]
            lemma, targ, pred = digraphify(lemma), digraphify(targ), digraphify(pred)
            rule = getEditClass(lemma, targ)
            # print("".join(lemma), "\t", "".join(targ), "\t", 
            #       "".join(pred), "".join(rule), ruleType(rule), correct)

            rtp = tuple(ruleType(rule))
            #record overall plural distribution
            plurTypes[rtp] += 1
            #memorize examples for printouts
            typeExes[rtp].append((lemma, targ))
            typeExes[rule].append((lemma, targ))

            #memorize accuracy by rule
            accByClass[rtp].append(correct)

            #memorize accuracy by specific rule within overall type
            rulesByType[rtp][rule].append(correct)

            #what rule did the system actually apply
            producedRule = getEditClass(lemma, pred)
            ptp = tuple(ruleType(producedRule))
            #record overall distr. of the output
            actualOutputs[ptp] += 1

            if "".join(rule) == "*+t*":
                specificRuleConf["".join(producedRule)] += 1

    # prints out confusion matrix row for the designated rule
    # print("Specific rule confusion:")
    # tot = 0
    # for predR, ct in specificRuleConf.most_common(20):
    #     print(predR, ct)
    #     tot += ct
    # print("others", sum(specificRuleConf.values()) - tot)
    # print()


    print("Total", tscore, "/", total, "\tOverall acc:", tscore/total)
    row["dev.score"] = "%.3f" % (tscore / total)

    for rtp in sorted(accByClass, key=lambda xx: np.mean(accByClass[xx])):
        #prints formal classes sorted by accuracy
        ex = typeExes[rtp][0]
        abc = sum(accByClass[rtp])
        num = len(accByClass[rtp])
        rstr = "".join([xx.replace("RULE_", "") for xx in rtp])
        print(rstr, "\t", abc, "/", num, "%.2f" % (abc / num), "".join(ex[0]), "~", "".join(ex[1]))

        rstr = rstr.rstrip("_").lower()
        row[rstr] = "%.3f" % (abc / num)

    specifics = {}
    specifics["lang"] = lang
    specifics["train.size"] = tsize
    specifics["run"] = run

    print("BY RULE TYPE:")
    for rtp, sub in rulesByType.items():
        #prints the same counts, organized into macroclasses
        print(rtp)
        for rule, outcomes in sub.items():
            abc = sum(outcomes)
            num = len(outcomes)
            specifics["".join(rule)] = abc / num

        for rule, outcomes in sorted(sub.items(), key=lambda xx: len(xx[1]), reverse=True)[:5]:
            #prints the microclasses under each macro
            abc = sum(outcomes)
            num = len(outcomes)
            ex = typeExes[rule][0]
            print("".join(rule), "".join(ex[0]), "~", "".join(ex[1]), 
                  "\t", abc, "/", num, "%.2f" % (abc/num))
        print()


    print()

    #doesn't do anything as far as I can see

    # print("CLASS:")

    # for rtp in sorted(accFeats, key=lambda xx: np.mean(accFeats[xx])):
    #     ex = typeExes[rtp][0]
    #     abc = sum(accFeats[rtp])
    #     num = len(accFeats[rtp])
    #     rstr = "".join([xx.replace("RULE_", "") for xx in rtp])
    #     print(rstr, "\t", abc, "/", num, "%.2f" % (abc / num), "".join(ex[0]), "~", "".join(ex[1]))

    print("ACTUAL:")
    num = sum(actualOutputs.values())
    for rtp, count in reversed(actualOutputs.most_common()):
        rstr = "".join([xx.replace("RULE_", "") for xx in rtp])
        print(rstr, "\t", count, "/", num, "%.2f" % (count / num))

        rstr = rstr.rstrip("_").lower()
        if "pref" in rstr:
            continue
        row["output." + rstr] = "%.3f" % (count / num)

    return row, specifics

if __name__ == "__main__":
    dd = sys.argv[1]
    data = []
    spec = []
    for subd in os.listdir(dd):
        print(subd)
        if os.path.isdir(dd + "/" + subd):
            row, specifics = scoreDir(dd + "/" + subd)
            data.append(row)
            spec.append(specifics)

    ofh = open("transfer_stats.csv", "a")
    writer = csv.DictWriter(ofh,
                            ["lang", "run", "train.size", "dev.score",
                             "suff", "stem", "stem_suff", "output.suff",
                             "output.stem", "output.stem_suff",
                             "output.syncretic"],
                            delimiter="\t")
    writer.writeheader()
    for row in data:
        writer.writerow(row)

    ofh = open("specific_stats.csv", "a")
    flds = ["lang", "run", "train.size"]
    for si in spec:
        for fld in si:
            if fld not in flds:
                flds.append(fld)
    writer = csv.DictWriter(ofh,
                            flds,
                            delimiter="\t")
    writer.writeheader()
    for row in spec:
        writer.writerow(row)
