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

def writeFile(dName, genericJob, replacements, jobName=None):
    if jobName is None:
        jobName = replacements["<NAME>"]
    fName = "%s/%s.sh" % (dName, jobName)

    print("writing", fName)

    with open(fName, "w") as ofh:
        with open(genericJob, "r") as ifh:
            for line in ifh:
                for rkey, rval in replacements.items():
                    line = line.replace(rkey, rval)
                ofh.write(line)

if __name__ == "__main__":
    genericExtract = "generic-extract.sh"
    genericRetrain = "generic-graph-retrain.sh"
    data = sys.argv[1]
    odir = sys.argv[2]

    os.makedirs("%s/stages" % odir, exist_ok=True)
    os.makedirs("%s/stageJobs" % odir, exist_ok=True)

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

    for stage in range(5):
        os.makedirs("%s/stageJobs/stage%d" % (odir, stage), exist_ok=True)
        os.makedirs("%s/stages/stage%d" % (odir, stage), exist_ok=True)

        for lang, fam in langFamily.items():
            if stage == 0:
                model = "%s/fam/%s/model0/checkpoints" % (odir, os.path.basename(fam))
            else:
                model = "%s/stages/stage%d/retrain-%s/model0/checkpoints" % (odir, stage - 1, os.path.basename(fam))

            replacements = {
                "<NAME>" : "extract-%s" % os.path.basename(lang).replace(".trn", ""),
                "<TRAIN>" : lang,
                "<ROOT>" : "%s/stages/stage%d" % (odir, stage),
                "<MODEL>" : model
                }
            writeFile("%s/stageJobs/stage%d" % (odir, stage),
                      genericExtract,
                      replacements)

        for fam in families:
            replacements = {
                "<NAME>" : "retrain-%s" % os.path.basename(fam),
                "<TRAIN>" : fam,
                "<ROOT>" : "%s/stages/stage%d" % (odir, stage),
                "<MODEL>" : model,
                "<GRAPH>" : "%s/stages/stage%d/graphs" % (odir, stage),
                }

            writeFile("%s/stageJobs/stage%d" % (odir, stage),
                      genericRetrain,
                      replacements)
