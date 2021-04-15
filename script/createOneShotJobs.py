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

from create2020Jobs import *

if __name__ == "__main__":
    oneshotCreateDataset = "oneshot-createdataset.sh"

    data = sys.argv[1]
    run = sys.argv[2]

    odir = run
    os.makedirs("%s/jobs/oneshot" % odir, exist_ok=True)
    os.makedirs("%s/oneshot" % odir, exist_ok=True)

    os.makedirs("%s/jobs/oneshot" % odir, exist_ok=True)
    os.makedirs("%s/oneshot" % odir, exist_ok=True)

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

    for df in dfiles:
        langName = os.path.basename(df).replace(".trn", "")
        famName = os.path.basename(os.path.dirname(df)).lower()

        famPar = "%s/fam/%s" % (odir, famName)
        if not os.path.exists(famPar):
            famPar = odir

        writeFile("%s/jobs/oneshot" % odir, df, oneshotCreateDataset, dParent=famPar, run=run,
                  jobName=langName, test=True)
