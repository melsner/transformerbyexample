from __future__ import division, print_function
import sys
import re
from collections import defaultdict
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

from writeOutput import *

if __name__ == "__main__":
    prd = sys.argv[1]

    ofn = os.path.dirname(prd) + "/variable-cases.txt"

    nCorrect = 0
    total = 0
    variableItems = 0
    variableItemsCorrect = 0
    decs = []
    with open(prd) as prdFh:
        with open(ofn, 'w') as ofh:
            for src, targ, pred in readPreds(prdFh):
                correct = (targ == pred)
                #print(targ, pred, correct)
                if correct:
                    nCorrect += 1
                total += 1
                decs.append((src, targ, pred, correct))
                if len(decs) == 5:
                    if caseVaries(decs):
                        variableItems += 1

                        for ((src, targ, pred, correct)) in decs:
                            correct = str(int(correct))
                            src, lang = re.split("TRG_", src)
                            ofh.write("%s\t%s\t%s\n" % (src, correct, lang))

                    decs = []

    print("Variable items: %d/%d = %.2g" % (variableItems, total, variableItems/total))
    print("Score: %d/%d = %.2g" % (nCorrect, total, nCorrect/total))
    print("Score within variable cases: %d/%d = %.2g" % (variableItemsCorrect, variableItems, variableItemsCorrect/variableItems))

