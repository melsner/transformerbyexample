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

from Seq2seq import seq2seq_runner, dataloader
from Seq2seq import model as model_lib

from s2sFlags import *
from utils import edist_alt, edist
from byexample import Data

def writeDevFiles(run, dpath):
    assert(os.path.isdir(dpath))
    devByLang = {}
    dataByLang = {}

    for root, dirs, files in os.walk(dpath):
        for fi in files:
            lang = fi
            valid = False
            for code in [".dev", ".trn"]:
                if code in lang:
                    lang = lang.replace(code, "")
                    valid = True

            if not valid:
                continue

            print("Reading", fi, "for", lang)

            rawData = np.loadtxt(root + "/" + fi, dtype=str, delimiter="\t")
            if fi.endswith(".dev"):
                devByLang[lang] = rawData
            else:
                dataByLang[lang] = rawData

    for lang, raw in dataByLang.items():
        data = Data(raw, lang=lang, nExemplars=args.n_exemplars)
        dataByLang[lang] = data
        rawDev = devByLang[lang]
        data.devSet = [(lemma, form, set(feats.split(";")), lang) for (lemma, form, feats) in rawDev]

        devpath = "%s/%s" % (run, lang)
        os.makedirs(devpath, exist_ok=True)
        devfile = "%s/dev.txt" % devpath
        if not os.path.exists(devfile):
            data.writeInstances(devfile, dev=True)

    return list(dataByLang.keys())

def runDev(run, langs, args, cpt):
    for lang in langs:
        workdir = "%s/%s/infer" % (run, lang)
        flags = S2SFlags(args, workdir)
        flags.train = None
        flags.dev = "%s/%s/dev.txt" % (run, lang)
        flags.checkpoint_to_restore = cpt
        print("Running", lang)
        seq2seq_runner.run(flags)

if __name__ == "__main__":
    args = get_arguments()

    run = args.run
    dpath = args.data

    langs = writeDevFiles(run, dpath)

    variant = 0
    workdir = "%s/model%d" % (run, variant)
    while os.path.exists(workdir):
        variant += 1
        workdir = "%s/model%d" % (run, variant)

    print("Looking for checkpoint")
    cpdir = os.path.abspath("%s/model%d/checkpoints/" % (run, variant - 1))
    cpt = None
    for fi in os.listdir(cpdir):
        if fi.endswith(".index"):
            cpt = cpdir + "/" + fi
    assert(cpt is not None)
    cpt.replace(".index", "")
    print("Checkpoint", cpt)

    runDev(run, langs, args, cpt)
