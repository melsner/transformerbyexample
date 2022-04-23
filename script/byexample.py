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

from Seq2seq import seq2seq_runner, dataloader
from Seq2seq import model as model_lib

from s2sFlags import *
from utils import edist_alt, edist, getEditClass, cacheWipe, get_size, findLatestModel
import classifyVariability
import editGraphs
import ruleFeatures

def createDataFromPath(dpath, args):
    allData = []
    for root, dirs, files in os.walk(dpath):
        for fi in files:
            lang = fi
            valid = False
            family = None

            if lang.endswith(".dev") or lang.endswith(".trn"):
                #set lang fam for 2020
                family = os.path.basename(root)

            for code in ["-train-low", "-train-high", ".trn"]:
                if code in lang:
                    lang = lang.replace(code, "")
                    valid = True

            if not valid:
                continue
                    
            print("Reading", fi, "for", lang, "in", family)

            rawData = np.loadtxt(root + "/" + fi, dtype=str, delimiter="\t")
            allData.append((rawData, lang, family))
                
    data = Data(allData, lang=None, family=None, nExemplars=args.n_exemplars, useEditClass=args.edit_class)
    return data

class Data:
    def __init__(self, instances, lang, family, nExemplars=1, useEditClass=False):
        self.lang = lang
        self.useEditClass = useEditClass
        self.frequencies = {}
        self.langProbs = {}

        if self.useEditClass:
            self.internEC = defaultdict(dict)
            self.revECTab = defaultdict(dict)
        if self.lang is not None:
            self.langFamilies = { self.lang : family }
            if instances.shape[1] == 4:
                print("Loading frequency data from last column of tsv.")
                self.frequencies[lang] = {}
                for (lemma, form, feats, freq) in instances:
                    try:
                        self.frequencies[lang][form] = float(freq)
                    except ValueError:
                        #print("No frequency for", lang, lemma)
                        self.frequencies[lang][form] = 1                        

                instances = instances[:, :-1]
                
            self.instances = [(lemma, form, set(feats.split(";")), lang, family) for (lemma, form, feats) in instances]
        else:
            self.langFamilies = {}
            self.instances = []
            for (raw, lang, fam) in instances:
                self.langFamilies[lang] = fam
                if raw.shape[1] == 4:
                    print("Loading frequency data from last column of tsv (%s)." % lang)
                    self.frequencies[lang] = Counter()
                    for (lemma, form, feats, freq) in raw:
                        try:
                            self.frequencies[lang][form] = float(freq)
                        except ValueError:
                            #print("No frequency for", lang, lemma)
                            self.frequencies[lang][form] = 1                        

                    raw = raw[:, :-1]

                local = [(lemma, form, set(feats.split(";")), lang, fam) for (lemma, form, feats) in raw]
                self.instances += local
        self.byFeature = defaultdict(list)
        self.byEditClass = defaultdict(lambda: defaultdict(list))
        self.allChars = set()
        if self.useEditClass:
            print("Compiling edit classes for", len(self.instances), "instances")
        for ind, (lemma, form, feats, lang, fam) in enumerate(self.instances):
            self.byFeature[frozenset(feats), lang].append((lemma, form, feats, lang))
            if self.useEditClass:
                if ind % 10000 == 0:
                    print("\t", ind, "...")
                editClass = self.getEditClass(feats, lang, lemma, form)
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
        for (lemma, form, feats, lang, fam) in self.instances:
            self.byFeature[frozenset(feats), lang].append((lemma, form, feats, lang))
            if self.useEditClass:
                editClass = self.getEditClass(feats, lang, lemma, form)
                self.byEditClass[frozenset(feats), lang][editClass].append((lemma, form, feats, lang))

    def filter(self, tag):
        self.instances = [(lemma, form, feats, lang, fam) for (lemma, form, feats, lang, fam) in self.instances if tag in feats]

    def conformToCharset(self, charset):
        newInstances = []
        for (lemma, form, feats, lang, fam) in self.instances:
            if not all([tt in charset for tt in lemma]):
                print("Lemma", lemma, "fails because", set(lemma).difference(charset))
            elif not all([tt in charset for tt in form]):
                print("Form", form, "fails because", set(form).difference(charset))
            else:
                newInstances.append((lemma, form, feats, lang, fam))
        self.instances = newInstances

    def __len__(self):
        return len(self.instances)

    def getEditClass(self, feats, lang, lemma, form):
        feats = frozenset(feats)
        ec = getEditClass(lemma, form)
        ifl = self.internEC[feats, lang]
        if ec not in ifl:
            ind = len(ifl)
            ifl[ec] = ind
            self.revECTab[feats, lang][ind] = ec

        return ifl[ec]

    def revEC(self, feats, lang, ind):
        feats = frozenset(feats)
        assert((feats, lang) in self.revECTab)
        return self.revECTab[feats, lang][ind]

    def precomputeClassSimilarity(self, verbose=1):
        self.similarClasses = {}
        printed = 0
        for (feats, lang), classes in self.internEC.items():
            print("Precomputing similarity over", feats, lang, "with", len(classes), "edit classes")
            print("size of sims tbl", get_size(self.similarClasses))
            sims = np.zeros((len(classes), len(classes)), dtype="int32")
            self.similarClasses[feats, lang] = sims
            for ci, cii in classes.items():
                for cj, cji in classes.items():
                    score = edist(ci, cj)
                    sims[cii, cji] = score
    
                if printed < verbose:
                    print("Class:", ci)
                    for sj in np.argsort(sims[cii]):
                        print("\t", self.revEC(feats, lang, sj), sims[cii, sj])
                    printed += 1

    def getNNExemplars(self, srcLemma, feats, lang, fam, nn, avoid=None, nSamples=100, nExemplars=5, verbose=False):
        available = self.byFeature[frozenset(feats), lang]
        available = [(lemma, form, feats, lang) for (lemma, form, feats, lang) in available if lemma != avoid]

        if len(available) == 0:
            raise ValueError("No exemplar (nn mode): lang %s feats %s" % (lang, str(feats)))

        nSamples = min(nSamples, len(available))
        choices = np.random.choice(len(available), size=nSamples, replace=False)
        insts = []
        for ind in choices:
            exLemma, exForm, exFeats, exLang = available[ind]
            src = "%s:%s>%s" % (srcLemma, exLemma, exForm)
            features = ruleFeatures.featureFn(lang, fam, feats, None, None, False)
            inst = " ".join(list(src) + features)
            insts.append((inst, available[ind]))

        scores = nn.scores([inst for (inst, ex) in insts])
        if verbose:
            for ((inst, ex), sc) in zip(insts, scores):
                print("\t", inst, "->", sc)

        scored = sorted(zip(insts, scores), key=lambda xx: xx[1], reverse=True)
        result = scored[:nExemplars]

        if verbose:
            for (inst, ex), sc in result:
                print("\t*", inst, "*->*", sc)

        result = [ex for ((inst, ex), sc) in result]
        return result

    def getExemplar(self, feats, lang, avoid=None, similar=None, similarityRank=False):
        assert(similarityRank in ["exact", "approximate", None])
        available = []
        if similar is not None and similarityRank == "exact":
            available = self.byEditClass[frozenset(feats), lang][similar]
            available = [(lemma, form, feats, lang) for (lemma, form, feats, lang) in available if lemma != avoid]

            # if available:
            #     print("Found", available[0], "for edit class", similar)

            if not available:
                edByCell = self.byEditClass[frozenset(feats), lang]
                edClasses = list(edByCell.keys())
                np.random.shuffle(edClasses)
                for edc in sorted(edClasses, key=lambda xx: edist(self.revEC(feats, lang, xx), 
                                                                  self.revEC(feats, lang, similar))):
                    #print("\tNext class", edc)

                    available = edByCell[edc]
                    available = [(lemma, form, feats, lang) for (lemma, form, feats, lang) in available if lemma != avoid]
                    if available:
                        break

        elif similar is not None and similarityRank == "approximate":
            #print("Getting approx exe", feats, lang, similar)
            relevantClasses = self.similarClasses[frozenset(feats), lang]
            edByCell = self.byEditClass[frozenset(feats), lang]
            if similar > relevantClasses.shape[0]:
                #if it's a dev-only edit class, just find something vaguely similar-looking
                return self.getExemplar(feats, lang, avoid=avoid, similar=similar, similarityRank="exact")

            nEdits = np.sort(relevantClasses[similar, :])
            available = []
            for attempt in range(5):
                sample = np.random.beta(1, 10)
                index = int(np.round(nEdits.shape[0] * sample - .5))
                ned = nEdits[index]
                #print("Selecting a class with", ned, "edits")
                classes = []
                for ii in range(nEdits.shape[0]):
                    if relevantClasses[similar, ii] == ned:
                        classes.append(ii)
                #print("Choosing from", classes)
                edc = classes[np.random.choice(len(classes))]
                #print("Chose", self.revEC(feats, lang, edc))
                if edc not in edByCell:
                    continue
                available = edByCell[edc]
                available = [(lemma, form, feats, lang) for (lemma, form, feats, lang) in available if lemma != avoid]
                #print(len(available), "options in this class")
                if available:
                    break

        else:
            available = self.byFeature[frozenset(feats), lang]
            available = [(lemma, form, feats, lang) for (lemma, form, feats, lang) in available if lemma != avoid]

        if len(available) == 0:
            if hasattr(self, "revECTab"):
                strSim = self.revEC(feats, lang, similar)
            else:
                strSim = "None"
            raise ValueError("No exemplar: lang %s feats %s similar %s mode %s" % (lang, str(feats), strSim, similarityRank))

        ri = np.random.choice(len(available))
        ex = available[ri]
        #print("Result is", ex)
        return ex

    def getGraphExemplar(self, feats, lang, similar, avoid=None):
        dists = self.graphDists[frozenset(feats), lang]
        strSim = self.revEC(feats, lang, similar)
        if strSim in dists:
            cellDist = dists[strSim]
            #print("for", feats, lang, "found", cellDist)
            prs = list(cellDist.items())
            prsOnly = [xx[1] for xx in prs]
            sample = np.random.choice(range(len(prsOnly)), p=prsOnly)
            selected = prs[sample][0]
            #print("chose", selected)
            #if it's not in the table, it can't be used, so just fall back on the original class
            selected = self.internEC[frozenset(feats), lang].get(selected, similar)

        else:
            selected = similar

        return self.getExemplar(feats, lang, similar=selected, avoid=avoid, similarityRank="exact")

    def langExemplars(self, limit, langSize):
        if langSize > limit:
            return 1
        else:
            return int(limit // langSize)

    def targProb(self, lang, form, feats, balance):
        assert(self.frequencies)
        if (lang, balance) in self.langProbs:
            return self.langProbs[lang, balance][form]
        else:
            self.langProbs[lang, balance] = {}

            formCounts = Counter()
            for (li, fi, fts, lg, fam) in self.instances:
                if lg == lang:
                    formCounts[fi] += 1

            ruleCounts = Counter()
            if balance == "rule":
                for (li, fi, fts, lg, fam) in self.instances:
                    if lg == lang:
                        rule = getEditClass(li, fi)
                        ruleCounts[rule] += 1
                                
            allStats = []
            for (li, fi, fts, lg, fam) in self.instances:
                if lg == lang:
                    stat = self.frequencies[lang][fi] / formCounts[fi]
                    if balance == "rule":
                        rule = getEditClass(li, fi)
                        stat = (1 / formCounts[fi]) * (1 / ruleCounts[rule])
                allStats.append(stat)

            allStats = np.array(allStats)
            
            if balance == "freq" or balance == "rule":
                allStats /= np.sum(allStats)
            elif balance == "logfreq":
                allStats = np.log(1 + allStats)
                allStats /= np.sum(allStats)
            else:
                top, topk = balance
                assert(top == "top")
                aInds = np.argsort(allStats)
                cutoff = allStats[aInds[-topk]]
                allStats = (allStats >= cutoff).astype(float)
                allStats /= np.sum(allStats)
                
            for (li, fi, fts, lg, fam), pr in zip(self.instances, allStats):
                self.langProbs[lang, balance][fi] = pr

            return self.langProbs[lang, balance][form]                

    def sampleTargets(self, instances, limit, nExemplars, balance):
        instPerLang = defaultdict(int)
        langSize = defaultdict(int)
        for (lemma, form, feats, lang, fam) in instances:
            langSize[lang] += 1

        #basic use case: just return all the targets in order
        #basic case with limit: return up to *limit* examples per language
        #dynamic: return up to *limit* examples, with *langExemplars* exes each
        #cases with "all" exemplars: return None as nExemplars
        #add balancing strategies (by freq, log freq, top-k)                    
        if limit is None:
            assert(nExemplars != "dynamic")
            assert(balance is None)
            for (lemma, form, feats, lang, fam) in instances:
                nx = nExemplars
                yield (lemma, form, feats, lang, fam, nx)

        else:
            assert(nExemplars != "all")
            for (lemma, form, feats, lang, fam) in instances:
                if instPerLang[lang] > limit:
                    continue

                if nExemplars == "dynamic":
                    nx = self.langExemplars(limit, langSize[lang])
                    totalSize = limit
                else:
                    nx = nExemplars
                    totalSize = min(limit, nx * langSize[lang])

                if balance is not None:
                    pr = self.targProb(lang, form, feats, balance)
                    if pr == 0:
                        continue
                    nx = int(pr * totalSize)
                    if nx == 0:
                        nx = int(np.random.random() < (pr * totalSize))

                    # print("item with prob", pr, "expected to get", (pr * totalSize), "exs",
                    #       "and gets", nx)

                    if nx == 0:
                        continue
                        
                instPerLang[lang] += nx
                yield (lemma, form, feats, lang, fam, nx)

    def exemplarDistribution(self, similarityRank, feats, lang, lemma, form, allowSelfExemplar=False,
                             balance=None):
        #cases to support:
        #random exemplars
        #similar exemplars
        #sampled exemplars later
        #hooks for approximate/mixed/graph/nn exemplar selection-- do not support, no plans to use these modes going forward

        if similarityRank in ["graph", "nn", "mixed", "approximate"]:
            assert(0), "Currently unsupported exemplar mode %s" % similarityRank

        assert(similarityRank in ["exact", None])

        #first, check if we need the edit class; if so, fetch
        similar = None
        if similarityRank == "exact":
            similar = self.getEditClass(feats, lang, lemma, form)

        if allowSelfExemplar:
            avoid = lemma
        else:
            avoid = None

        available = []
        if similar is not None and similarityRank == "exact":
            available = self.byEditClass[frozenset(feats), lang][similar]
            available = [(lemma, form, feats, lang) for (lemma, form, feats, lang) in available if lemma != avoid]

            # if available:
            #     print("Found", available[0], "for edit class", similar)

            if not available:
                edByCell = self.byEditClass[frozenset(feats), lang]
                edClasses = list(edByCell.keys())
                np.random.shuffle(edClasses)
                for edc in sorted(edClasses, key=lambda xx: edist(self.revEC(feats, lang, xx), 
                                                                  self.revEC(feats, lang, similar))):
                    #print("\tNext class", edc)

                    available = edByCell[edc]
                    available = [(lemma, form, feats, lang) for (lemma, form, feats, lang) in available if lemma != avoid]
                    if available:
                        break

        else:
            available = self.byFeature[frozenset(feats), lang]
            available = [(lemma, form, feats, lang) for (lemma, form, feats, lang) in available if lemma != avoid]

        if len(available) == 0:
            if hasattr(self, "revECTab"):
                strSim = self.revEC(feats, lang, similar)
            else:
                strSim = "None"
            raise ValueError("No exemplar: lang %s feats %s similar %s mode %s" % (lang, str(feats), strSim, similarityRank))

        prs = None
        if balance is not None:
            prs = [self.targProb(lang, form, feats, balance) for (lemma, form, feats, lang) in available]
            prs = np.array(prs)
            if np.sum(prs) > 0:
                prs /= np.sum(prs)
            else:
                prs = np.ones_like(prs)
                prs /= np.sum(prs)

        return available, prs
            
    def writeInstances(self, ofn, dev=False, allowSelfExemplar=False, limit=None, useSimilarExemplar=None, exemplarNN=None, extraFeatures=False, 
                       balanceTargets=None, balanceExemplars=None):
        assert(not (dev and allowSelfExemplar))
        assert(not (useSimilarExemplar and exemplarNN))
        #will allow this for some dev runs and see what happens, but be careful not to use for eval scores
        #assert(not (dev and useSimilarExemplar))
        if dev:
            instances = self.devSet
            limit = None
            nExemplars = 5
            balanceTargets = None
        else:
            instances = self.instances
            nExemplars = self.nExemplars

        with open(ofn, "w") as ofh:
            for ind, (lemma, form, feats, lang, fam, nx) in enumerate(self.sampleTargets(instances, limit, nExemplars, 
                                                                                         balance=balanceTargets)):
                if ind % 1000 == 0:
                    print(ind, "/", len(instances), "instances written...")

                try:
                    exes, exePrs = self.exemplarDistribution(useSimilarExemplar, feats, lang, lemma, form,
                                                             balance=balanceExemplars, allowSelfExemplar=allowSelfExemplar)
                except ValueError as err:
                    print("Singleton feature vector", feats, lemma)
                    print("Detailed error", err)
                    continue

                if self.nExemplars == "all":
                    size = len(exes)
                    replace = False
                else:
                    size = nx
                    replace = True

                samples = np.random.choice(len(exes), p=exePrs, size=size, replace=replace)
                samples = [exes[ii] for ii in samples]
                for (exLemma, exForm, exFeats, exLang) in samples:
                    src = "%s:%s>%s" % (lemma, exLemma, exForm)
                    targ = form
                    features = ruleFeatures.featureFn(lang, fam, feats, 
                                                      getEditClass(lemma, form), 
                                                      getEditClass(exLemma, exForm),
                                                      extraFeatures)
                    ofh.write("%s\t%s\t%s\n" % (src, targ, ";".join(features)))
                    if extraFeatures in ["all", "rule"]:
                        ofh.write(ruleFeatures.classificationInst(src, targ, features))
            
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

def headFile(infile, outfile, nlines):
    with open(infile) as ifh:
        with open(outfile, "w") as ofh:
            for ind, line in enumerate(ifh):
                ofh.write(line)
                if ind > nlines:
                    break

if __name__ == "__main__":
    #load the data
    print("Script start")
    args = get_arguments()
    print("Arguments parsed")
    run = args.run
    dfile = args.data

    data = None

    exemplarNN = None
    if args.exemplar_nn is not None:
        nnFilePath = findLatestModel(args.exemplar_nn)
        variant = 0
        exScratch = os.path.abspath(args.exemplar_nn + "../../scratch%d" % variant)
        while os.path.exists(exScratch):
            variant += 1
            exScratch = os.path.abspath(args.exemplar_nn + "../../scratch%d" % variant)
        print("Creating exemplar nn scratch dir", exScratch)
        os.makedirs(exScratch)
        flags = S2SFlags(args, exScratch + "/model")
        exTrain = os.path.abspath(args.exemplar_nn + "../../train.txt")
        exDev = os.path.abspath(args.exemplar_nn + "../../dev.txt")
        print("Setting train and dev to", exTrain, exDev)
        headFile(exTrain, exScratch + "/train.txt", 100)
        headFile(exDev, exScratch + "/dev.txt", 100)
        flags.train = exScratch + "/train.txt"
        flags.dev = exScratch + "/dev.txt"
        flags.checkpoint_to_restore = findLatestModel(args.load_other)
        exemplarNN = classifyVariability.buildAndLoadModel(flags, nnFilePath)

        print("---Successful load of exemplar nn---")

    trainExists = os.path.exists("%s/train.txt" % run)

    if not trainExists:

        if os.path.isdir(dfile):
            data = createDataFromPath(dfile, args)
        else:
            lang = os.path.basename(dfile)
            family = None
            if lang.endswith(".dev") or lang.endswith(".trn"):
                #set lang fam for 2020
                family = os.path.basename(os.path.dirname(dfile))
                family = family.lower()

            for code in ["-dev", "-test", "-train-low", "-train-high", ".trn", ".dev"]:
                lang = lang.replace(code, "")

            print("Running single dataset for", lang, "in", family)
            rawData = np.loadtxt(dfile, dtype=str, delimiter="\t")
            data = Data(rawData, lang=lang, family=family, nExemplars=args.n_exemplars, useEditClass=args.edit_class)

    if not trainExists and not args.generate_file:
        if not args.devset:
            data.splitDev(instances=2000)
        else:
            lang = os.path.basename(args.devset)
            devFamily = os.path.basename(os.path.dirname(args.devset))
            for code in ["-dev", "-test", "-train-low", "-train-high", ".trn", ".dev", ".tst"]:
                lang = lang.replace(code, "")
            if devFamily != "GOLD-TEST":
                family = devFamily
            family = family.lower()
            print("Identified devset as", lang, family)

            rawDev = np.loadtxt(args.devset, dtype=str, delimiter="\t")[:, :3]
            data.devSet = [(lemma, form, set(feats.split(";")), lang, family) for (lemma, form, feats) in rawDev]

    assert(args.edit_class != "graph" or args.edit_graph is not None)
    if args.edit_class == "graph":
        graphDists = editGraphs.loadGraphs(data, args.edit_graph)
        data.graphDists = graphDists

    if args.generate_file:
        os.makedirs(run, exist_ok=True)
        data.writeInstances("%s/dev.txt" % run, allowSelfExemplar=args.allow_self_exemplar, limit=args.limit_train,
                            useSimilarExemplar=args.edit_class, exemplarNN=exemplarNN, 
                            extraFeatures=args.extra_features, balanceTargets=args.balance_targets, balanceExemplars=args.balance_exemplars)
        sys.exit(0)

    if not trainExists:
        print(len(data), "instances")
        if args.edit_class == "approximate":
            data.precomputeClassSimilarity(verbose=5)

        os.makedirs(run, exist_ok=True)
        if args.junk:
            data.junkInstances("%s/train.txt" % run)
            data.junkInstances("%s/dev.txt" % run, dev=True)
        else:
            data.writeInstances("%s/train.txt" % run, allowSelfExemplar=args.allow_self_exemplar,
                                limit=args.limit_train,
                                useSimilarExemplar=args.edit_class, exemplarNN=exemplarNN,
                                extraFeatures=args.extra_features,
                                balanceTargets=args.balance_targets, balanceExemplars=args.balance_exemplars)
            data.writeInstances("%s/dev.txt" % run, dev=True, useSimilarExemplar=args.edit_class,
                                exemplarNN=exemplarNN, extraFeatures=args.extra_features,
                                balanceTargets=args.balance_targets, balanceExemplars=args.balance_exemplars)

    if args.load_other:
        args.load_other = findLatestModel(args.load_other)

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

        if args.append_train and not trainExists:
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
        best = 0
        for fi in os.listdir(cpdir):
            if fi.endswith(".index"):
                cptN = int(fi.replace(".index", "").split("-")[1])
                if cptN > best:
                    cpt = cpdir + "/" + fi
                    best = cptN
        assert(cpt is not None)
        cpt.replace(".index", "")
        print("Checkpoint", cpt)
        flags.checkpoint_to_restore = cpt

    if data is not None:
        del data

    print("Starting run")
    seq2seq_runner.run(flags)
