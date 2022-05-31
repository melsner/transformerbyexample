from __future__ import division, print_function
import sys
from collections import defaultdict
import re
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

from create2020Jobs import writeFile

if __name__ == "__main__":
    genericDerived = "generic-derived-2020.sh"
    genericTest = "generic-testmodel-2020.sh"
    ubm = "ubm_challenge_22_trajectory"

    data = sys.argv[1]
    run = sys.argv[2]

    odir = run
    os.makedirs("%s/jobs" % odir, exist_ok=True)
    os.makedirs("%s/fine" % odir, exist_ok=True)

    dfiles = []
    for root, dirs, files in os.walk(data):
        for fi in files:
            if fi.endswith(".trn") or fi.endswith(".train"):
                dfile = root + "/" + fi
                dfiles.append(dfile)

    for df in dfiles:
        #classifier learning stack: by language
        langName = os.path.basename(df).replace(".trn", "").replace(".train", "")
        basicLang = os.path.dirname(df) + "/" + re.sub("_([0-9]+)", "", langName)
        famName = "None"

        writeFile("%s/jobs" % odir, df, genericDerived, dev=basicLang + ".dev", 
                  dParent=ubm, run=run)

        writeFile("%s/jobs" % odir, df, genericTest, 
                  dev=basicLang + ".test",
                  dParent="%s/fine/%s" % (odir, langName), run=run, 
                  test=True,
                  jobName="%s-test" % langName)
