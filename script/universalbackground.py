from __future__ import division, print_function
import sys
from collections import defaultdict, Counter
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
import unicodedata

from Seq2seq import seq2seq_runner, dataloader
from Seq2seq import model as model_lib

from s2sFlags import *

from byexample import *

def writeInstances(ofn, instances, extraFeatures=False):
    with open(ofn, "w") as ofh:
        for (lemma, targ, exLemma, exForm, code) in instances:
            src = "%s:%s>%s" % (lemma, exLemma, exForm)
            features = ruleFeatures.featureFn(code[0], code[1],
                                              [],
                                              getEditClass(lemma, targ),
                                              getEditClass(exLemma, exForm),
                                              extraFeatures)
            ofh.write("%s\t%s\t%s\n" % (src, targ, ";".join(features)))

def fakeInflection(cset, lemma, ex):
    aLengthD = [ (1, .2), (2, .2), (3, .2), (4, .2), (5, .2) ]
    iLengthD = [ (1, .5), (2, .5) ]
    pref = sampledStr(cset, aLengthD)
    suff = sampledStr(cset, aLengthD)
    inf = sampledStr(cset, iLengthD)

    exForm = ex
    targ = lemma
    code = []
    if np.random.random() < .1:
        rr = np.random.choice(min(len(targ), len(exForm)))
        exForm = exForm[:rr] + inf + exForm[rr:]
        rr = np.random.choice(len(targ))
        targ = targ[:rr] + inf + targ[rr:]
        code.append("infix")
    if np.random.random() < .5:
        exForm = pref + exForm
        targ = pref + targ
        code.append("prefix")
    if np.random.random() < .5:
        exForm = exForm + suff
        targ = targ + suff
        code.append("suffix")

    if code == []:
        code.append("suffix")

    code = "-".join(code)

    return targ, exForm, (code, "synthetic")

def discoverVocab(dpath):
    charSets = defaultdict(Counter)
    allFeats = Counter()
    lemmaLengths = []
    formLengths = []
    knownFiles = set()
    families = {}
    for root, dirs, files in os.walk(dpath):
        for fi in files:
            if fi.endswith("test-covered") or fi.endswith(".tst"):
                continue
            if fi in knownFiles:
                continue

            knownFiles.add(fi)
            lang = fi
            family = None
            valid = False
            for code in ["-dev", "-test", "-train-low", "-train-high", ".dev", ".trn"]:
                if code in lang:
                    if lang.endswith(".dev") or lang.endswith(".trn"):
                        #set lang fam for 2020
                        family = os.path.basename(root)
                        family = family.lower()

                    lang = lang.replace(code, "")
                    families[lang] = family
                    valid = True

            if not valid:
                continue

            dfile = root + "/" + fi

            print("Reading", dfile, "for", lang, "in", family)

            rawData = np.loadtxt(dfile, dtype=str, delimiter="\t")
            charSet = Counter()
            for (lemma, form, feats) in rawData:
                if len(form) > 50 and " " in form:
                    print("Anomalous word form (skipping)", fi, form)
                    continue

                nbsp = unicodedata.lookup("NO-BREAK SPACE")
                if nbsp in lemma or nbsp in form:
                    print("Non-breaking space for space:", fi, lemma, form)
                lemma = lemma.replace(nbsp, " ")
                form = form.replace(nbsp, " ")

                charSet.update(lemma)
                charSet.update(form)
                lemmaLengths.append(len(lemma))
                formLengths.append(len(form))

                allFeats.update(feats.split(";"))

            charSets[lang].update(charSet)

    maxLemma = max(lemmaLengths) #np.percentile(lemmaLengths, cutoff)
    maxForm = max(formLengths) #np.percentile(formLengths, cutoff)

    return charSets, families, allFeats, maxLemma, maxForm, lemmaLengths, formLengths

def syntheticInstances(num, charSets, families, feats, strLengthD, extraFeatures=False):
    res = []

    #generate a "word" containing each valid char, in blocks of 10
    for lang, chs in charSets.items():
        allCh = "".join(list(chs.keys()))
        for ind in np.arange(0, len(allCh), 10):
            lemma = allCh[ind: ind + 10]
            targ = lemma
            exLemma = lemma
            exForm = lemma
            res.append((lemma, targ, exLemma, exForm, (lang, families[lang])))

    #generate a "word" containing each valid feature, in blocks of 10
    if extraFeatures:
        allFeats = list(feats)

        allFeats += ["TRG_RULE_PREF", "TRG_RULE_SUFF", "TRG_RULE_STEM", "TRG_RULE_SUPPLETIVE"]
        for tp in ["PREF", "SUFF", "IN"]:
            for mod in ["_RM", ""]:
                for op in ["TRG_CMP_%s_COPY",
                           "TRG_CMP_%s_REPLACE",
                           "TRG_CMP_%s_CONTACT",
                           "TRG_CMP_%s_DISTANT"]:
                    allFeats += op % (tp + mod)

        for ind in np.arange(0, len(allFeats), 10):
            lemma = "a"
            targ = "a"
            langCode = lang + ";" + ";".join(["TRG_CELL_%s" % xx for xx in allFeats[ind: ind + 10]])
            res.append((lemma, targ, lemma, targ, (langCode, families[lang])))

    langs = list(charSets.keys())
    for ii in range(num):
        lang = np.random.choice(langs)
        cset = charSets[lang]
        lemma = sampledStr(cset, strLengthD)
        exLemma = sampledStr(cset, strLengthD)
        targ, exForm, code = fakeInflection(cset, lemma, exLemma)
        res.append((lemma, targ, exLemma, exForm, code))

    return res

def sampledStr(charSet, strLengthD):
    chars = list(charSet.items())
    cdist = np.array([xx[1] for xx in chars], dtype="float64")
    cdist /= np.sum(cdist)

    strLengthDV = np.array([xx[1] for xx in strLengthD])
    rr = np.random.multinomial(1, strLengthDV)
    am = np.argmax(rr)
    nChars = strLengthD[am][0]


    res = []
    for ii in range(int(nChars)):
        rr = np.random.multinomial(1, cdist)
        am = np.argmax(rr)
        res += chars[am][0]

    return "".join(res)

if __name__ == "__main__":
    args = get_arguments()

    run = args.run
    dpath = args.data

    charSets, families, feats, maxLemma, maxForm, lemmaLengths, formLengths = discoverVocab(dpath)

    print(len(charSets), "charsets")
    for lang, chs in charSets.items():
        print(lang, "".join([xx[0] for xx in chs.most_common(5)]))
    print(len(feats), "feats")
    print(feats)
    print("longest lemma", maxLemma)
    print("longest form", maxForm)

    strLengthD = Counter(lemmaLengths).most_common()
    norm = sum([xx[1] for xx in strLengthD])
    strLengthD = [(kk, vv / norm) for (kk, vv) in strLengthD]
    print("String lengths:")
    print(strLengthD)

    #override length settings
    if args.src_length == -1:
        #length of an instance is lemma + lemma + form + a few control chars
        #number of extra chars is set to 5 as a fudge factor
        args.src_length = int(5 + 2 * maxLemma + maxForm)
    if args.targ_length == -1:
        args.targ_length = int(5 + maxForm)

    instances = syntheticInstances(11000, charSets, families, feats, strLengthD, 
                                   extraFeatures=args.extra_features)
    devSet = instances[-1000:]
    trainSet = instances[:-1000]
    os.makedirs(run, exist_ok=True)
    writeInstances("%s/train.txt" % run, trainSet, extraFeatures=args.extra_features)
    writeInstances("%s/dev.txt" % run, devSet, extraFeatures=args.extra_features)

    print("Parsing flags")
    variant = 0
    workdir = "%s/model%d" % (run, variant)
    while os.path.exists(workdir):
        variant += 1
        workdir = "%s/model%d" % (run, variant)
    flags = S2SFlags(args, workdir)
    flags.train = "%s/train.txt" % run
    flags.dev = "%s/dev.txt" % run

    if variant > 0:
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
