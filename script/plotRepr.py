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

    tsneMat = sklearn.manifold.TSNE(perplexity=5).fit_transform(wMat)

    plt.scatter(tsneMat[:, 0], tsneMat[:, 1])
    for ind, li in enumerate(langLst):
        plt.text(tsneMat[ind, 0], tsneMat[ind, 1], li)

    plt.savefig("results/%s-langs.png" % os.path.basename(os.path.dirname(sys.argv[1])))
