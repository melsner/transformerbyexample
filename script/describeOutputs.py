import sys
import os
import re
import csv

from collections import defaultdict, Counter
from byexample import getEditClass
from classifyVariability import readPreds, readCases

def skeleton(form, vowels):
    res = []
    for ci in form:
        if ci in vowels:
            res.append("V")
        else:
            res.append("C")
    return "".join(res)

if __name__ == "__main__":
    fpath = sys.argv[1]

    with open(fpath) as fh:
        preds = list(readPreds(fh))

    with open(fpath + ".csv", "w") as ofh:
        writer = csv.DictWriter(ofh, ["lemma", "affix", "exemplar", 
                                      "targ", "pred", "rule", "lang", "fam", "correct", "lemmaCV", "affixCV"])

        writer.writeheader()

        for (src, targ, pred, correct) in preds:
            features = re.findall("TRG_([A-Z]*_[a-z]*)", src)
            content = re.sub("TRG_([A-Z]*_[a-z]*)", "", src)
            lemma, ex, exForm = re.match("(.*):(.*)>(.*)", content).groups()
            #print(features, lemma, ex, exForm, targ, pred, correct)
            rule = str(getEditClass(pred, targ))
            #print(rule)
            affix = targ.replace(lemma, "")

            lemmaCV = skeleton(lemma, "aeiou")
            affixCV = skeleton(affix, "aeiou")
            lang = None
            fam = None
            for fi in features:
                if fi.startswith("LANG_"):
                    lang = fi.replace("LANG_", "")
                if fi.startswith("FAM_"):
                    fam = fi.replace("FAM_", "")

            writer.writerow(
                {"lemma" : lemma, "affix" : affix, "exemplar" : ex, "targ" : targ, "pred" : pred, "rule" : rule, "lang" : lang,
                 "fam" : fam, "correct" : correct, "lemmaCV" : lemmaCV, "affixCV" : affixCV } )

