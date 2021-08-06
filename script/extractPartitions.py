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
from byexample import Data, getEditClass, createDataFromPath
from classifyEditUtil import *

def computeScores(model, data, cell, nSamples=10, verbose=False):
    result = nx.DiGraph()
    sub = data.byEditClass[cell]
    for edCl in sub:
        words = data.byEditClass[cell][edCl]
        csize = len(words)
        # if csize < 10:
        #     #nb: does not actually skip anything
        #     if verbose:
        #         print("Skipping class", "".join(data.revEC(cell[0], cell[1], edCl)), "size:", csize)

        sample = [words[xx] for xx in np.random.choice(len(words), nSamples)]

        xxScore = []
        for ii in range(2):
            sample2 = sampleExes(words, sample)
            instances, targets = formatInstances(sample, sample2)
            scores = getScores(instances, targets, model)
            #print("Sample instance", instances[0].replace(" ", ""), "-->", targets[0].replace(" ", ""))
            #print(scores[0])
            scores = np.sum(scores, axis=1)
            #print(scores[0])
            #ev = model.translate([instances[0]])
            #print("Predicted output:", ev)
            xxScore.append(np.mean(scores))

        mscore = np.mean(xxScore)
        sd = np.std(xxScore)

        if verbose:
            print("".join(data.revEC(cell[0], cell[1], edCl)), "size:", csize,
                  "self score:", np.mean(xxScore), "std:", sd)

            print()
            print()

        if sd > 0:
            clScores = []
            for edCl2 in sub:
                words2 = data.byEditClass[cell][edCl2]
                if len(words2) < 10:
                    continue

                sample2 = sampleExes(words2, sample)
                instances, targets = formatInstances(sample, sample2)
                scores = getScores(instances, targets, model)
                #print("Sample instance", instances[0].replace(" ", ""),
                #      "-->", targets[0].replace(" ", ""))
                example = (instances[0].replace(" ", ""), targets[0].replace(" ", ""))
                # print(scores[0])
                # print("shape of sc0", scores[0].shape)
                # t0 = targets[0].replace(" ", "") + "X" * 77
                # for ind, si in enumerate(scores[0]):
                #     print(si, t0[ind])
                # print()

                scores = np.sum(scores, axis=1)
                #print(scores[0])
                #ev = model.translate([instances[0]])
                #print("Predicted output:", ev)
                sc = np.mean(scores)
                value = (sc - mscore) / sd
                clScores.append((edCl2, example, sc, value))

            clScores.sort(key=lambda xx: xx[-1])

            if verbose:
                for edCl2, ex, sc, value in clScores:
                    print("\t", "".join(data.revEC(cell[0], cell[1], edCl2)), ex[0], "-->", ex[1],
                          "\t\t", value, sc)
                print()

            for edCl2, ex, sc, value in clScores:
                n1 = data.revEC(cell[0], cell[1], edCl)
                n2 = data.revEC(cell[0], cell[1], edCl2)

                result.add_edge(n1, n2, loss=sc, sdev=value, weight=np.exp(-value))

    return result

def getScores(instances, targets, model):
    reprs = model.prepare_for_forced_validation(instances, model.src_language_index)
    targReprs = model.prepare_for_forced_validation(targets, model.trg_language_index)
    targInp = targReprs[:, :-1]
    targReal = targReprs[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = model_lib.create_masks(reprs, targInp)
    predictions, _ = model.transformer.call(reprs, targInp, False, enc_padding_mask, combined_mask,
                                    dec_padding_mask)

    mask = tf.math.logical_not(tf.math.equal(targReal, 0))
    loss_ = model.loss_object(targReal, predictions)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return loss_
    #scores = model.loss_function(targReal, predictions)
    #return scores

def formatInstances(targets, exemplars):
    insts = []
    targs = []
    for ti, xi in zip(targets, exemplars):
        exLemma, exForm, exFeats, exLang = xi
        srcLemma, srcForm, srcFeats, srcLang = ti
        src = "%s:%s>%s" % (srcLemma, exLemma, exForm)
        src = src.replace(" ", "_")
        srcFam = data.langFamilies[srcLang]
        inst = " ".join(list(src) + ["TRG_LANG_%s" % srcLang, "TRG_FAM_%s" % srcFam])
        targ = srcForm
        targ.replace(" ", "_")
        targ = " ".join(list(targ))
        insts.append(inst)
        targs.append(targ)
    return insts, targs

def buildModel(flags):
  hparams, flags = seq2seq_runner.handle_preparation_flags(flags)

  # Prepare data.
  all_data = seq2seq_runner.prepare_data(flags, hparams)
  trg_language_index = all_data.trg_language_index
  trg_feature_index = all_data.trg_feature_index
  trg_max_len_seq = all_data.trg_max_len_seq
  trg_max_len_ft = all_data.trg_max_len_ft
  split_sizes = all_data.split_sizes

  # Get model.
  model = model_lib.Model(hparams, all_data, flags)
  cpName = hparams.checkpoint_to_restore.replace("ckpt-", "manual_save").replace(".index", ".h5")
  model.transformer.load_weights(cpName)
  return model

def sampleExes(words, prevSamples):
    nSamples = len(prevSamples)
    samples = [words[xx] for xx in np.random.choice(len(words), nSamples)]
    if len(words) == 1:
        return samples

    for ii, si in enumerate(samples):
        if si == prevSamples[ii]:
            acceptable = [xx for xx in words if xx != prevSamples[ii]]
            xx = acceptable[np.random.choice(len(acceptable))]
            samples[ii] = xx

    return samples

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

        for code in ["-dev", "-test", "-train-low", "-train-high", ".trn", ".dev"]:
            lang = lang.replace(code, "")

        print("Running single dataset for", lang, "in", family)
        rawData = np.loadtxt(dfile, dtype=str, delimiter="\t")
        data = Data(rawData, lang=lang, family=family, nExemplars=args.n_exemplars, 
                    useEditClass=args.edit_class)

    #load model
    os.makedirs(run, exist_ok=True)
    os.makedirs(run + "/graphs", exist_ok=True)
    scratchDir = run + "/scratch"
    os.system("rm -rf %s" % scratchDir)
    flags = S2SFlags(args, scratchDir)
    flags.train = args.load_other + "/../../train.txt"
    flags.dev = args.load_other + "/../../dev.txt"
    args.load_other = findLatestModel(args.load_other)
    print("Restoring", args.load_other)
    flags.checkpoint_to_restore = args.load_other

    model = buildModel(flags)

    #separate them into edit classes
    #--should be done automatically by edit_class option

    for cell, sub in data.byEditClass.items():
        print("Cell:", cell)
        print([data.revEC(cell[0], cell[1], xx) for xx in sub])
        print()

    #create the matrix of class/class scores
    for cell, sub in data.byEditClass.items():
        print()
        print("############# scores for %s %s ################" % cell)
        print()

        graph = computeScores(model, data, cell, verbose=True)
        cellName = "-".join(list(cell[0])) + "-" + cell[1]
        #nx.write_graphml(graph, "%s/graphs/%s.gml" % cellName)
        pkl.dump(graph, open("%s/graphs/%s.pkl" % (run, cellName), "wb"))
