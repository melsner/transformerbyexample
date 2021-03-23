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
    import cPickle as pickle
else:
    import pickle

import sklearn.manifold
from matplotlib import pyplot as plt

if __name__ == "__main__":
    wMat, langLst = pickle.load(open(sys.argv[1], "rb"))

    lpath = "../task0-data/DEVELOPMENT-LANGUAGES"
    langFam = {}
    for root, dirs, files in os.walk(lpath):
        for fi in files:
            if fi.endswith(".dev"):
                fi = fi.replace(".dev", "")
                langFam[fi] = os.path.basename(root)

    print(langFam)

    devLgs = []
    devLabels = []
    for ind, li in enumerate(langLst):
        if "prefix" in li or "suffix" in li or "infix" in li:
            langFam[li] = "synthetic"

        if li in langFam:
            devLgs.append(wMat[ind])
            devLabels.append(li)

    devlgs = np.array(devLgs)
    wMat = devLgs
    langLst = devLabels

    families = list(set(langFam.values()))

    cmap = plt.get_cmap("Dark2")

    tsneMat = sklearn.manifold.TSNE(perplexity=2).fit_transform(wMat)

    scByFam = {}

    plt.figure(figsize=(10, 10))
    for ind, li in enumerate(langLst):
        plt.text(tsneMat[ind, 0], tsneMat[ind, 1], li)
        scByFam[langFam[li]] = plt.scatter(tsneMat[ind, 0], tsneMat[ind, 1], color=cmap(families.index(langFam[li])))

    plt.legend([scByFam[fi] for fi in families], families)

    plt.savefig("results/%s-langs.png" % os.path.basename(os.path.dirname(sys.argv[1])))
