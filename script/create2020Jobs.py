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

def writeFile(dName, train, genericJob, dParent, run, jobName=None, location="fine", test=False, special=None, limitTrain=None):
    langName = os.path.basename(train).replace(".trn", "")
    if jobName is None:
        jobName = langName

    fName = "%s/%s.sh" % (dName, jobName)

    print("writing", fName)
    dev = train.replace(".trn", ".dev")
    if test:
        dev = train.replace(".trn", ".tst")
        dev = os.path.dirname(train) + "/../../GOLD-TEST/" + os.path.basename(dev)
    parent = dParent + "/" + "/model0/checkpoints/"
    #exnn = run + "/fine/" + langName + "-classify-variable/model0/checkpoints/"
    famName = os.path.basename(os.path.abspath(dParent))
    exnn = run + "/fam/" + famName + "-learn-class/model0/checkpoints/"

    if special is None:
        special = "ERROR"

    print("\t", langName, train, dev, parent)
    with open(fName, "w") as ofh:
        with open(genericJob, "r") as ifh:
            for line in ifh:
                line = line.replace("<NAME>", jobName)
                line = line.replace("<TRAIN>", train)
                line = line.replace("<DEV>", dev)
                line = line.replace("<PARENT>", parent)
                line = line.replace("<ROOT>", run)
                line = line.replace("<EXNN>", exnn)
                line = line.replace("<LOC>", location)
                line = line.replace("<SPECIAL>", special)
                line = line.replace("<LIMIT>", str(limitTrain))
                ofh.write(line)

if __name__ == "__main__":
    genericDerived = "generic-derived-2020.sh"
    genericFamily = "generic-family-2020.sh"
    genericWriteSelection = "generic-writeselect.sh"
    genericEval = "generic-eval.sh"
    genericLearnClassifier = "generic-learnclassifier.sh"
    genericCreateDataset = "generic-createdataset.sh"
    genericDerivedSelected = "generic-derived-selected.sh"
    genericFamilyNN = "generic-family-nn.sh"
    genericTest = "generic-testmodel.sh"

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
        famName = os.path.basename(fam)
        writeFile("%s/jobs/fam" % odir, fam, genericFamily, dParent=run, run=run, location="fam")
        #classifier learning stack: by family
        writeFile("%s/jobs/fam" % odir, fam, genericWriteSelection, dParent="%s/fam/%s" % (odir, famName),
                  run=run, jobName="%s-write-select" % famName, location="fam")
        selectDev = "%s/fam/%s-write-select/dev.txt" % (odir, famName)
        writeFile("%s/jobs/fam" % odir, selectDev, genericEval, dParent="%s/fam/%s" % (odir, famName), run=run, 
                  jobName="%s-eval" % famName, location="fam")
        devOut = "%s/fam/%s-eval/%s-write-select-dev.txt/predictions_dev.txt" % (odir, famName, famName)
        writeFile("%s/jobs/fam" % odir, devOut, genericLearnClassifier, dParent="%s/fam/%s" % (odir, famName), 
                  run=run, jobName="%s-learn-class" % famName, location="fam")

        writeFile("%s/jobs/fam" % odir, fam, genericFamilyNN, dParent="%s/fam/%s" % (odir, famName), 
                  run=run, location="fam", jobName="%s-post-nn" % famName)

    for df in dfiles:
        #writeFile("%s/jobs/fine" % odir, df, genericDerived, dParent="%s/fam/%s" % (odir, famName), run=run)

        #classifier learning stack: by language
        langName = os.path.basename(df).replace(".trn", "")
        famName = os.path.basename(os.path.dirname(df))
        #writeFile("%s/jobs/fine" % odir, df, genericWriteSelection, dParent="%s/fam/%s" % (odir, famName),
        #          run=run, jobName="%s-write-select" % langName)
        #selectDev = "%s/fine/%s-write-select/dev.txt" % (odir, langName)
        #writeFile("%s/jobs/fine" % odir, selectDev, genericEval, dParent="%s/fam/%s" % (odir, famName), run=run, 
        #          jobName="%s-eval" % langName)
        #devOut = "%s/fine/%s-eval/%s-write-select-dev.txt/predictions_dev.txt" % (odir, langName, langName)
        #writeFile("%s/jobs/fine" % odir, devOut, genericLearnClassifier, dParent="%s/fam/%s" % (odir, famName), 
        #          run=run, jobName="%s-classify-variable" % langName)
        writeFile("%s/jobs/fine" % odir, df, genericCreateDataset, dParent="%s/fam/%s" % (odir, famName), run=run,
                  jobName="%s-create-dataset" % langName)

        writeFile("%s/jobs/fine" % odir, df, genericDerived, dParent="%s/fam/%s" % (odir, famName), run=run)

        writeFile("%s/jobs/fine" % odir, df, genericTest, dParent="%s/fine/%s" % (odir, langName), run=run, test=True,
                  jobName="%s-test" % langName)
