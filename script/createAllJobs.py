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
    import cPickle as pkl
else:
    import pickle as pkl

def findFile(dd, suff):
    if not os.path.exists(dd):
        for sub in os.listdir(os.path.dirname(dd)):
            if sub.startswith(os.path.basename(dd)):
                dd = os.path.dirname(dd) + "/" + sub
                break

    for fi in os.listdir(dd):
        if fi.endswith(suff):
            return dd + "/" + fi

def writeFile(dName, langName, genericJob, dParent=None):
    fName = dName + "/" + os.path.basename(langName) + ".sh"
    name = os.path.basename(langName) + "_ml"
    print("writing", fName)
    if "--" not in langName:
        train = findFile(langName, "-train-high")
        dev = "NONE"
        parent = "NONE"
    else:
        train = findFile(langName, "-train-low")
        dev = findFile(langName, "-dev")
        parLang = os.path.basename(langName).split("--")[0]
        parent = dParent + "/" + "%s_ml" % parLang + "/model0/checkpoints/"

    print("\t", name, train, dev, parent)
    with open(fName, "w") as ofh:
        with open(genericJob, "r") as ifh:
            for line in ifh:
                line = line.replace("<NAME>", name)
                line = line.replace("<TRAIN>", train)
                line = line.replace("<DEV>", dev)
                line = line.replace("<PARENT>", parent)
                ofh.write(line)

if __name__ == "__main__":
    genericParent = "generic-parent.sh"
    genericDerived = "generic-derived.sh"
    data = sys.argv[1]
    odir = sys.argv[2]

    os.makedirs("%s/parent" % odir, exist_ok=True)
    os.makedirs("%s/derived" % odir, exist_ok=True)

    dfiles = os.listdir(data)
    parents = set()

    for df in dfiles:
        parent = df.split("--")[0]
        parents.add(parent)

        writeFile("%s/derived" % odir, data + "/" + df, genericDerived, dParent="runs/parent")

    for par in parents:
        writeFile("%s/parent" % odir, data + "/" + par, genericParent)
