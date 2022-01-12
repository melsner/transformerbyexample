import sys
import os
import re
import csv
import numpy as np

from collections import defaultdict, Counter
from utils import readPreds, getEditClass
from outputsByClass import *

if __name__ == "__main__":
    preds = open(sys.argv[1])
    classPreds = open(sys.argv[2])
    keys = open(sys.argv[3])
    judgements = open(sys.argv[4])

    prs = []
    for (src, targ, prob, correct) in readPreds(classPreds):
        prs.append(prob)

    bySource = defaultdict(list)
    for (src, targ, pred, correct), key, prob in zip(readPreds(preds), keys, prs):
        srcLemma, exe = src.split(":")
        bySource[srcLemma].append((exe, targ, pred, correct, key, prob))

    srcToRating = defaultdict(list)
    for line in judgements:
        (src, form, feats, rating) = line.strip().split("\t")
        src = "".join(src.split())
        form = "".join(form.split())
        rating = float(rating)
        srcToRating[src].append((form, feats, rating))

    ofh = open(os.path.dirname(sys.argv[3]) + "/covered.csv", "w")
    writer = csv.DictWriter(ofh, ["lemma", "form", "rating", "centered.rating", "found", "rules", "supporting.words", "pr0", "pr1", "pr2", "pr3", "pr4", "med.pr", "rank.byprob", "rank.bysupport", "most.freq.support",
                                  "most.freq.support0", "pct.top.10", "pct.top.25", "pct.top.50", "pct.top.100"])
    writer.writeheader()
    coverage = 0
    total = 0
    for src, forms in srcToRating.items():
        outToStats, predToTotal = statsByOutput(bySource[src])
        allForms = list(outToStats.keys())
        rankedForms = list(sorted(allForms, key=lambda xx: predToTotal[xx], reverse=True))
        rankedFormsPr = list(
            sorted(allForms,
                   key=
                   lambda xx: min([min(prs)
                                   for (rule, nWords, wMax, ex, prs) in outToStats[xx]])))

        print(src)
        print("By support:", rankedForms)
        print("By prob:", rankedFormsPr)

        meanRating = np.mean([rating for (form, feats, rating) in forms])

        unifiedPrs = []
        for (form, feats, rating) in forms:
            if form in outToStats:
                stats = outToStats[form]
                for (rule, nWords, wMax, ex, prs) in stats:
                    for pr in prs:
                        unifiedPrs.append((pr, form))
        unifiedPrs.sort(key=lambda xx: xx[0])

        for (form, feats, rating) in forms:
            total += 1
            if form in outToStats:
                stats = outToStats[form]
                coverage += 1
                found = 1
                rules = len(stats)
                support = predToTotal[form]
                allPrs = []
                maxSupport = 0
                support0 = 0

                for (rule, nWords, wMax, ex, prs) in stats:
                    if wMax > maxSupport:
                        maxSupport = wMax
                    if len(allPrs) == 0 or min(prs) < min(allPrs):
                        support0 = wMax
                    allPrs += prs

                allPrs = sorted(allPrs)
                medPr = np.median(allPrs)
                while len(allPrs) < 5:
                    allPrs.append(1)
                pr0, pr1, pr2, pr3, pr4 = allPrs[:5]

                rankProb = rankedFormsPr.index(form)
                rankSupport = rankedForms.index(form)

                cutoffPcts = []
                for cutoff in [10, 25, 50, 100]:
                    unifCut = unifiedPrs[:cutoff]
                    nConsistent = len([(pr, predictedForm) for (pr, predictedForm) in unifCut
                                       if predictedForm == form])
                    pct = nConsistent / len(allPrs)
                    cutoffPcts.append(pct)

                print(src, "~", form, rankProb)

            else:
                found = 0
                rules = 0
                support = 0
                pr0, pr1, pr2, pr3, pr4 = (0, 0, 0, 0, 0)
                medPr = 0
                print(src, "~", form, "not found")
                rankProb = len(rankedFormsPr)
                rankSupport = len(rankedForms)
                cutoffPcts = [0, 0, 0, 0]
                maxSupport = 0
                support0 = 0

            writer.writerow({"lemma": src, "form": form,
                             "rating": rating, "centered.rating": rating - meanRating,
                             "found": found, "rules": rules, "supporting.words": support,
                             "pr0": pr0, "pr1": pr1, "pr2": pr2, "pr3": pr3, "pr4": pr4,
                             "med.pr": medPr, 
                             "rank.byprob":rankProb, "rank.bysupport":rankSupport,
                             "most.freq.support": maxSupport, "most.freq.support0": support0,
                             "pct.top.10": cutoffPcts[0],
                             "pct.top.25": cutoffPcts[1],
                             "pct.top.50": cutoffPcts[2],
                             "pct.top.100": cutoffPcts[3],
                         })

    print("Coverage:", coverage, "/", total, ";\t", coverage / total)
