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

def getEditClass(lemma, form):
    cost, (alt1, alt2) = edist_alt(lemma, form)
    #print("Aligned", lemma, form, cost, alt1, alt2)
    alt = []
    ap1 = 0
    ap2 = 0
    while ap1 < len(alt1) or ap2 < len(alt2):
        while ap1 < len(alt1) and alt1[ap1][1] == False:
            alt.append("-%s" % lemma[ap1])
            ap1 += 1
        while ap2 < len(alt2) and alt2[ap2][1] == False:
            alt.append("+%s" % form[ap2])
            ap2 += 1

        if ap1 < len(alt1) and ap2 < len(alt2) and alt1[ap1][1] == True and alt2[ap2][1] == True:
            alt.append("-")

        while ap1 < len(alt1) and ap2 < len(alt2) and alt1[ap1][1] == True and alt2[ap2][1] == True:
            ap1 += 1
            ap2 += 1

    #print("Edit class:", lemma, form, alt)
    return tuple(alt)

class Data:
    def __init__(self, instances, lang, nExemplars=1, useEditClass=False):
        self.lang = lang
        self.useEditClass = useEditClass
        if self.lang is not None:
            self.instances = [(lemma, form, set(feats.split(";")), lang) for (lemma, form, feats) in instances]
        else:
            self.instances = []
            for (raw, lang) in instances:
                local = [(lemma, form, set(feats.split(";")), lang) for (lemma, form, feats) in raw]
                self.instances += local
        self.byFeature = defaultdict(list)
        self.byEditClass = defaultdict(lambda: defaultdict(list))
        self.allChars = set()
        for (lemma, form, feats, lang) in self.instances:
            self.byFeature[frozenset(feats), lang].append((lemma, form, feats, lang))
            if self.useEditClass:
                editClass = getEditClass(lemma, form)
                self.byEditClass[frozenset(feats), lang][editClass].append((lemma, form, feats, lang))
            self.allChars.update(form)
        self.devSet = []
        self.nExemplars = nExemplars

    def splitDev(self, instances=None, fraction=None):
        if fraction is not None:
            instances = int(len(self) * fraction)

        inds = np.arange(0, len(self))
        np.random.shuffle(inds)
        devInds = inds[:instances]

        self.devSet = []
        for di in devInds:
            self.devSet.append(self.instances[di])

        newInstances = []
        for xi in inds[instances:]:
            newInstances.append(self.instances[xi])
        self.instances = newInstances
        self.byFeature = defaultdict(list)
        self.byEditClass = defaultdict(lambda: defaultdict(list))
        for (lemma, form, feats, lang) in self.instances:
            self.byFeature[frozenset(feats), lang].append((lemma, form, feats, lang))
            if self.useEditClass:
                editClass = getEditClass(lemma, form)
                self.byEditClass[frozenset(feats), lang][editClass].append((lemma, form, feats, lang))

    def filter(self, tag):
        self.instances = [(lemma, form, feats, lang) for (lemma, form, feats, lang) in self.instances if tag in feats]

    def conformToCharset(self, charset):
        newInstances = []
        for (lemma, form, feats, lang) in self.instances:
            if not all([tt in charset for tt in lemma]):
                print("Lemma", lemma, "fails because", set(lemma).difference(charset))
            elif not all([tt in charset for tt in form]):
                print("Form", form, "fails because", set(form).difference(charset))
            else:
                newInstances.append((lemma, form, feats, lang))
        self.instances = newInstances

    def __len__(self):
        return len(self.instances)

    def getExemplar(self, feats, lang, avoid=None, similar=None):
        available = []
        if similar:
            available = self.byEditClass[frozenset(feats), lang][similar]
            available = [(lemma, form, feats, lang) for (lemma, form, feats, lang) in available if lemma != avoid]

            # if available:
            #     print("Found", available[0], "for edit class", similar)

            if not available:
                edByCell = self.byEditClass[frozenset(feats), lang]
                edClasses = list(edByCell.keys())
                for edc in sorted(edClasses, key=lambda xx: edist(xx, similar)):
                    #print("\tNext class", edc)

                    available = edByCell[edc]
                    available = [(lemma, form, feats, lang) for (lemma, form, feats, lang) in available if lemma != avoid]
                    if available:
                        break

        else:
            available = self.byFeature[frozenset(feats), lang]
            available = [(lemma, form, feats, lang) for (lemma, form, feats, lang) in available if lemma != avoid]

        if len(available) == 0:
            raise ValueError("No exemplar")

        ri = np.random.choice(len(available))
        ex = available[ri]
        #print("Result is", ex)
        return ex

    def langExemplars(self, limit, langSize):
        if langSize > limit:
            return 1
        else:
            return int(limit // langSize)

    def writeInstances(self, ofn, dev=False, allowSelfExemplar=False, limit=None, useSimilarExemplar=False):
        assert(not (dev and allowSelfExemplar))
        assert(not (dev and useSimilarExemplar))
        if dev:
            instances = self.devSet
        else:
            instances = self.instances

        instPerLang = defaultdict(int)
        langSize = defaultdict(int)
        for (lemma, form, feats, lang) in instances:
            langSize[lang] += 1

        with open(ofn, "w") as ofh:
            for ind, (lemma, form, feats, lang) in enumerate(instances):
                if ind % 1000 == 0:
                    print(ind, "/", len(instances), "instances written...")

                if limit and instPerLang[lang] > limit:
                    continue

                if self.nExemplars == "all":
                    available = self.byFeature[frozenset(feats)]
                    available = [(exL, fL, ftL, lL) for (exL, fL, ftL, lL) in available if exL != lemma]
                    for exLemma, exForm, exFeats, exLang in available:
                        src = "%s:%s>%s" % (lemma, exLemma, exForm)
                        targ = form
                        ofh.write("%s\t%s\tLANG_%s\n" % (src, targ, lang))

                else:
                    editClass = None
                    if useSimilarExemplar:
                        editClass = getEditClass(lemma, form)
                        #print("Edit class", lemma, form, editClass)

                    if self.nExemplars == "dynamic":
                        if not dev:
                            nExemplars = self.langExemplars(limit, langSize[lang])
                        else:
                            nExemplars = 5
                    else:
                        nExemplars = self.nExemplars

                    for exN in range(nExemplars):
                        instPerLang[lang] += 1
                        try:
                            if allowSelfExemplar:
                                ex = self.getExemplar(feats, lang, similar=editClass)
                            else:
                                ex = self.getExemplar(feats, lang, avoid=lemma, similar=editClass)
                        except ValueError:
                            print("Singleton feature vector", feats, lemma)
                            continue
                            #raise

                        exLemma, exForm, exFeats, exLang = ex

                        src = "%s:%s>%s" % (lemma, exLemma, exForm)
                        targ = form
                        ofh.write("%s\t%s\tLANG_%s\n" % (src, targ, lang))

def shuffleData(data):
    inds = np.arange(data.shape[0])
    np.random.shuffle(inds)
    return data[inds, ...]

def loadVocab(vocab):
    srcVoc = set()
    targVoc = set()
    with open(vocab) as vfh:
        for line in vfh:
            flds = line.strip().split("\t")
            if flds[0] == "src_vocab":
                srcVoc.add(flds[2])
            elif flds[0] == "trg_vocab":
                targVoc.add(flds[2])

    if "_" in srcVoc:
        srcVoc.add(" ")

    if "_" in targVoc:
        targVoc.add(" ")

    return srcVoc, targVoc

if __name__ == "__main__":
    #load the data
    print("Script start")
    args = get_arguments()
    print("Arguments parsed")
    run = args.run
    dfile = args.data

    if not os.path.exists("%s/train.txt" % run):

        if os.path.isdir(dfile):
            allData = []
            for root, dirs, files in os.walk(dfile):
                for fi in files:
                    lang = fi
                    valid = False
                    for code in ["-dev", "-test", "-train-low", "-train-high", ".trn"]:
                        if code in lang:
                            lang = lang.replace(code, "")
                            valid = True

                    if not valid:
                        continue
                    
                    print("Reading", fi, "for", lang)

                    rawData = np.loadtxt(root + "/" + fi, dtype=str, delimiter="\t")
                    allData.append((rawData, lang))
                
            data = Data(allData, lang=None, nExemplars=args.n_exemplars, useEditClass=args.edit_class)
        else:
            lang = os.path.basename(dfile)
            for code in ["-dev", "-test", "-train-low", "-train-high"]:
                lang = lang.replace(code, "")
            print("Running single dataset for", lang)
            rawData = np.loadtxt(dfile, dtype=str, delimiter="\t")
            data = Data(rawData, lang=lang, nExemplars=args.n_exemplars, useEditClass=args.edit_class)

    if not os.path.exists("%s/dev.txt" % run):
        if not args.devset:
            data.splitDev(instances=2000)
        else:
            rawDev = np.loadtxt(args.devset, dtype=str, delimiter="\t")
            data.devSet = [(lemma, form, set(feats.split(";")), lang) for (lemma, form, feats) in rawDev]

    if args.generate_file:
        os.makedirs("dataset_%s" % run, exist_ok=True)
        data.writeInstances("dataset_%s/dev.txt" % run)
        sys.exit(0)

    if not os.path.exists("%s/train.txt" % run):
        print(len(data), "instances")
        os.makedirs(run, exist_ok=True)
        if args.junk:
            data.junkInstances("%s/train.txt" % run)
            data.junkInstances("%s/dev.txt" % run, dev=True)
        else:
            data.writeInstances("%s/train.txt" % run, allowSelfExemplar=args.allow_self_exemplar, limit=args.limit_train,
                                useSimilarExemplar=args.edit_class)
            data.writeInstances("%s/dev.txt" % run, dev=True)

    if args.load_other:

        if not args.load_other.endswith(".index"):
            print("Searching for latest checkpoint")
            best = 0
            bestC = None

            for cpt in os.listdir(args.load_other):
                if cpt.endswith(".index"):
                    cptN = int(cpt.replace(".index", "").split("-")[1])
                    if cptN > best:
                        best = cptN
                        bestC = cpt

            assert(bestC is not None)
            args.load_other += "/" + cpt
            print("Using", args.load_other)

    print("Parsing flags")
    variant = 0
    workdir = "%s/model%d" % (run, variant)
    while os.path.exists(workdir):
        variant += 1
        workdir = "%s/model%d" % (run, variant)
    flags = S2SFlags(args, workdir)
    flags.train = "%s/train.txt" % run
    flags.dev = "%s/dev.txt" % run

    #look for a checkpoint
    if args.load_other:
        print("Loading from pretrained checkpoint", args.load_other)
        flags.checkpoint_to_restore = os.path.abspath(args.load_other)

        if args.append_train:
            prevTrain = os.path.dirname(args.load_other) + "/../../train.txt"
            print("Appending previous training data", prevTrain)
            with open(flags.train, "a") as ofh:
                with open(prevTrain) as ifh:
                    for line in ifh:
                        ofh.write(line)

    elif variant > 0:
        print("Model dir exists, looking for checkpoint")
        cpdir = os.path.abspath("%s/model%d/checkpoints/" % (run, variant - 1))
        cpt = None
        for fi in os.listdir(cpdir):
            if fi.endswith(".index"):
                cpt = cpdir + "/" + fi
        assert(cpt is not None)
        cpt.replace(".index", "")
        print("Checkpoint", cpt)
        flags.checkpoint_to_restore = cpt

    print("Starting run")
    seq2seq_runner.run(flags)
