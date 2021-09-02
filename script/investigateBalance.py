import sys
import os
import re
import csv
import numpy as np

from collections import defaultdict, Counter
from utils import readPreds, getEditClass

from s2sFlags import *
from utils import edist_alt, edist, cacheWipe, get_size, findLatestModel
from byexample import *

import matplotlib.pyplot as plt

def getRuleCounts(targs, data, rules=None):
    counts = Counter()
    for (lemma, form, feats, lang, fam, nx) in targs:
        rule = getEditClass(lemma, form)
        counts[rule] += nx

    if rules is None:
        rules = sorted(list(counts.keys()), key=counts.get, reverse=True)
    else:
        for ri in counts:
            if ri not in rules:
                rules.append(ri)

    vals = []
    for ri in rules:
        vals.append(counts[ri])

    return vals, rules

def makePlot(data, mode, rules=None):
    basic = data.sampleTargets(data.instances, 20000, 5, mode)
    ruleCounts, rules = getRuleCounts(basic, data, rules)
    fig, ax = plt.subplots()
    ax.set_title("Mode %s" % str(mode))
    ax.bar(list(range(len(ruleCounts))), ruleCounts)
    txtRules = ["".join(xx) for xx in rules]
    ax.set_xticks(list(range(len(rules))))
    ax.set_xticklabels(txtRules, rotation=-45)
    return rules

if __name__ == "__main__":
    print("Script start")
    args = get_arguments()
    print("Arguments parsed")
    run = args.run
    dfile = args.data

    #read all the examples
    if os.path.isdir(dfile):
        data = createDataFromPath(dfile, args)
    else:
        lang = os.path.basename(dfile)
        family = None
        if lang.endswith(".dev") or lang.endswith(".trn"):
            #set lang fam for 2020
            family = os.path.basename(os.path.dirname(dfile))
            family = family.lower()

        for code in ["-dev", "-test", "-train-low", "-train-high", ".trn", ".dev", ".train"]:
            lang = lang.replace(code, "")

        print("Running single dataset for", lang, "in", family)
        rawData = np.loadtxt(dfile, dtype=str, delimiter="\t")
        data = Data(rawData, lang=lang, family=family, nExemplars=args.n_exemplars, 
                    useEditClass="exact")

    data.langFamilies = { "deu" : "germanic",
                          "nld" : "germanic",
                          "eng" : "germanic",
                          "dummy" : "debug" }

    print("Loaded data")

    rules = makePlot(data, None)
    rules = makePlot(data, "freq", rules)
    rules = makePlot(data, "logfreq", rules)
    #rules = makePlot(data, ("top", 100), rules)
    #rules = makePlot(data, ("top", 1000), rules)
    rules = makePlot(data, "rule", rules)
    plt.show()
