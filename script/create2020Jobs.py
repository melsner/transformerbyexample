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

def writeFile(dName, train, genericJob, dParent, run, jobType="learn"):
    langName = os.path.basename(train).replace(".trn", "")
    jobName = langName
    if jobType != "learn":
        jobName = "%s-%s" % (os.path.basename(dParent), jobType)
    fName = "%s/%s.sh" % (dName, jobName)
    print("writing", fName)
    dev = train.replace(".trn", ".dev")
    parent = dParent + "/" + "/model0/checkpoints/"

    print("\t", langName, train, dev, parent)
    with open(fName, "w") as ofh:
        with open(genericJob, "r") as ifh:
            for line in ifh:
                line = line.replace("<NAME>", jobName)
                line = line.replace("<TRAIN>", train)
                line = line.replace("<DEV>", dev)
                line = line.replace("<PARENT>", parent)
                line = line.replace("<ROOT>", run)
                ofh.write(line)

if __name__ == "__main__":
    genericDerived = "generic-derived-2020.sh"
    genericFamily = "generic-family-2020.sh"
    genericWriteSelection = "generic-writeselect.sh"
    genericEval = "generic-eval.sh"
    genericLearnClassifier = "generic-learnclassifier.sh"
    data = sys.argv[1]
    run = sys.argv[2]

    odir = run
    os.makedirs("%s/jobs/fine" % odir, exist_ok=True)
    os.makedirs("%s/jobs/fam" % odir, exist_ok=True)
    os.makedirs("%s/fine" % odir, exist_ok=True)
    os.makedirs("%s/fam" % odir, exist_ok=True)

    dfiles = []
    families = set()
    langFamily = {}
    for root, dirs, files in os.walk(data):
        for fi in files:
            if fi.endswith(".trn"):
                dfile = root + "/" + fi
                dfiles.append(dfile)
                families.add(root)
                langFamily[dfile] = root

    for fam in families:
        writeFile("%s/jobs/fam" % odir, fam, genericFamily, dParent=run, run=run)
        writeFile("%s/jobs/fam" % odir, fam, genericWriteSelection, dParent=run, run=run, jobType="write-select")
        famName = os.path.basename(fam)
        selectDev = "%s/fam/%s-write-select/dev.txt" % (odir, famName)
        writeFile("%s/jobs/fam" % odir, selectDev, genericEval, dParent="%s/fam/%s" % (odir, famName), run=run, jobType="eval")
        devOut = "%s/fam/%s-eval/%s-write-select-dev.txt/predictions_dev.txt" % (odir, famName, famName)
        writeFile("%s/jobs/fam" % odir, devOut, genericLearnClassifier, dParent="%s/fam/%s" % (odir, famName), 
                  run=run, jobType="learn-class")

    for df in dfiles:
        famName = os.path.basename(langFamily[df])
        #write a dataset subcreation job
        writeFile("%s/jobs/fine" % odir, df, genericDerived, dParent="%s/fam/%s" % (odir, famName), run=run)
