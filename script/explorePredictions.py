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

import networkx as nx

from Seq2seq import seq2seq_runner, dataloader
from Seq2seq import model as model_lib

from s2sFlags import *
from utils import edist_alt, edist, cacheWipe, get_size, findLatestModel
from byexample import *

def mostFrequentWord(data, edCl):
    words = data.byEditClass[fullCell][edCl]    
    frs = []
    for (lemma, form, feats, lang) in words:
        try:
            freq = data.frequencies[lang][form]
        except KeyError:
            freq = 0
        frs.append(freq)

    return max(frs)

if __name__ == "__main__":
    print("Script start")
    args = get_arguments()
    print("Arguments parsed")
    run = args.run
    dfile = args.data

    #read all the examples
    if os.path.isdir(dfile):
        data = createDataFromPath(dfile, args)
    else:
        lang = os.path.basename(dfile)
        family = None
        if lang.endswith(".dev") or lang.endswith(".trn"):
            #set lang fam for 2020
            family = os.path.basename(os.path.dirname(dfile))
            family = family.lower()

        for code in ["-dev", "-test", "-train-low", "-train-high", ".trn", ".dev", ".train"]:
            lang = lang.replace(code, "")

        print("Running single dataset for", lang, "in", family)
        rawData = np.loadtxt(dfile, dtype=str, delimiter="\t")
        #rawData = rawData[:, :3] #drop orthographic
        data = Data(rawData, lang=lang, family=family, nExemplars=args.n_exemplars, 
                    useEditClass="exact")

    data.langFamilies = { "deu" : "germanic",
                          "nld" : "germanic",
                          "eng" : "germanic",
                          "gle" : "celtic",
                          "isl" : "Icelandic",
                          "klr" : "", #how did this happen?
                          "sme" : "North_Sami",
                          "zul" : "romance" #placeholder for spanish
    }

    print("Loaded data")
    if args.no_exemplars:
        print("No exemplars mode active.")

    #srcLemma = "Adeliepinguin"
    #targ = "Adeliepinguine"
    #cell = (frozenset(["NOM", "PL", "N"]), "deu")
    nSamples = 10

    dev = args.devset
    workdir = run + "/judgements"
    lemmas = set()
    os.makedirs(workdir, exist_ok=True)
    with open(dev) as devfh:
        editFh = open(workdir + "/key.txt", "w")
        classFh = open(workdir + "/class.txt", "w")
        ofh = open(workdir + "/dev.txt", "w")

        for line in devfh:
            #(srcLemma, targ, cell, orthLemma, orthForm) = line.strip().split("\t")
            (srcLemma, targ, cell) = line.strip().split("\t")[:3]

            if args.nospace:
                srcLemma = srcLemma.replace(" ", "")
                targ = targ.replace(" ", "")

            cell = frozenset(cell.split(";"))
            if srcLemma in lemmas and not args.no_exemplars:
                continue

            lemmas.add(srcLemma)
            print("Writing examples for", srcLemma)
            dotInd = os.path.basename(dev).index(".")
            lang = os.path.basename(dev)[:dotInd]
            fam = data.langFamilies[lang]
            fullCell = (cell, lang)
            assert(fullCell in data.byEditClass)

            sub = data.byEditClass[fullCell]

            if args.no_exemplars:
                src = srcLemma

                features = ruleFeatures.featureFn(lang, fam,
                                                  cell,
                                                  getEditClass("", ""),
                                                  getEditClass("", ""),
                                                  extraFeatures=args.extra_features)
                ofh.write("%s\t%s\t%s\n" % (src, targ, ";".join(features)))
                classFh.write(ruleFeatures.classificationInst(src, targ, features))
            else:

                for edCl in sub:
                    mostFreq = mostFrequentWord(data, edCl)
                    words = data.byEditClass[fullCell][edCl]
    
                    if args.limit_train and len(words) <= args.limit_train:
                        continue
    
                    sample = [words[xx] for xx in 
                              np.random.choice(len(words), min(len(words), nSamples),
                                               replace=False)]
                    for xi in sample:
                        exLemma, exForm, exFeats, exLang = xi
                        if exLemma == srcLemma:
                            continue #no self exemplars
    
                        src = "%s:%s>%s" % (srcLemma, exLemma, exForm)
    
                        features = ruleFeatures.featureFn(lang, fam,
                                                          cell,
                                                          getEditClass(exLemma, exForm),
                                                          getEditClass(exLemma, exForm),
                                                          extraFeatures=args.extra_features)
                        ofh.write("%s\t%s\t%s\n" % (src, targ, ";".join(features)))
                        row = [getEditClass(exLemma, exForm), len(words), mostFreq,
                               "%s>%s" % (words[0][0], words[0][1])]
                        editFh.write("\t".join([str(xx) for xx in row]))
                        editFh.write("\n")
                        classFh.write(ruleFeatures.classificationInst(src, targ, features))
    
