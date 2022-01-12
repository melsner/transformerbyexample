from __future__ import division, print_function
import sys
from collections import defaultdict, Counter
import os
import numpy as np
import argparse
import random
import math
import six
import unicodedata
import re
import csv
if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl

from utils import getEditClass
from classifyVariability import readPreds, readCases

def classifyAlt(lemma, target, suffix="n"):
    #now we have two problems... sigh
    print("querying", lemma, target)
    vowels = "aeiouáéíóúăüö"
    v1 = re.search("([%s]+)([^%s]*)([aei])r$" % (vowels, vowels), lemma)
    print("matched", v1.groups(), lemma)
    consonants = v1.group(2)
    found = None
    if consonants == "":
        v2 = re.search("([%s]+)%s%s$" % (vowels, v1.group(3), suffix), target)
        if v2:
            found = v2.group(1)
    else:
        print("attempt match", "^%s.*?(%s)([%s]+)" % (suffix, "".join(reversed(consonants)), vowels), "".join(reversed(target)))
        v2 = re.search("^%s.*?(%s)([%s]+)" % (suffix, "".join(reversed(consonants)), vowels), "".join(reversed(target)))
        if v2:
            print("success, got", v2.groups())
            found = "".join(reversed(v2.group(2)))
        #v2 = re.search("([%s]+)(%s).*%s$" % (vowels, consonants, suffix), target)
    
    if not v2:
        print("fallback")
        v2 = re.search("([%s]+)[aeu]yen$" % (vowels,), target) #acensuar -> acensuayen, should be u~u
        if v2:
            found = v2.group(1)
        else:
            v2 = re.search("([%s]+)[aei]n$" % (vowels,), target) #estatuir -> estatuen, should be u~u
            if v2:
                found = v2.group(1)
            else:
                v2 = re.search("([%s]+)%s$" % (vowels, suffix), target) #huir -> hin, should match u~0
                if v2:
                    found = v2.group(1)

    if found:
        print("matched", v2.groups(), found, lemma)
        alt = v1.group(1), found
    else:
        print("warning:", lemma, target)
        alt = v1.group(1), None
    return alt

def stressShift(rule):
    stab = {
        "a" : "á",
        "e" : "é",
        "i" : "í",
        "o" : "ó",
        "u" : "ú"
    }
    return rule[0] in stab and rule[1] == stab[rule[0]]

if __name__ == "__main__":
    preds = open(sys.argv[1])
    classPreds = open(sys.argv[2])
    keys = open(sys.argv[3])
    of = sys.argv[4]
    of2 = sys.argv[5]

    #use these names
    names = {
        ('*', '-i', '-r', '+e', '+n'): "abatir",
        ('*', '+i', '*', '-r', '+n'): "acertar",
        ('*', '-i', '+í', '*', '-r', '+n'): "acuantiar",
        ('*', '+i', '*', '-i', '-r', '+e', '+n'): "adherir",
        ('*', '-u', '+ú', '*', '-r', '+n'): "acensuar",
        ('*', '-o', '+u', '+e', '*', '-r', '+n'): "absolver",
        ('*', '+i', '+ñ', '*', '-ñ', '-i', '-r', '+n'): "astreñir", #some alignment issues for this one
        ('*', '+i', '+g', '*', '-g', '-i', '-r', '+n'): "astreñir",
        ('*', '+i', '+d', '*', '-d', '-i', '-r', '+n'): "astreñir",
        ('*', '+i', '+t', '*', '-t', '-i', '-r', '+n'): "astreñir",
        ('*', '+i', '+c', '*', '-c', '-i', '-r', '+n'): "astreñir",
        ('*', '-i', '-r', '+y', '+e', '+n'): "abluir",
        ('*', '-r', '+n'): "abacorar"
    }

    prs = []
    for (src, targ, prob, correct) in readPreds(classPreds):
        prs.append(prob)

    bySource = defaultdict(list)
    byTarget = defaultdict(Counter)
    for (src, targ, pred, correct), key, prob in zip(readPreds(preds), keys, prs):
        srcLemma, exe = src.split(":")
        bySource[srcLemma].append((exe, targ, pred, correct, key, prob))

        exeRule, nWords, wMax, ruleEx = key.strip().split("\t")
        exeRule = eval(exeRule)
        exeRule = names.get(exeRule, None)
        if exeRule:
            byTarget[srcLemma, pred][exeRule] += 1

    outcomes = defaultdict(Counter)
    outcomesByEx = defaultdict(Counter)

    for src, data in bySource.items():
        for (exe, targ, pred, correct, key, prob) in data:
            srcRule = getEditClass(src, targ)
            exeRule, nWords, wMax, ruleEx = key.strip().split("\t")
            exeRule = eval(exeRule)
            srcRule = names.get(srcRule, None)
            exeRule = names.get(exeRule, None)
            if srcRule is None or exeRule is None:
                continue

            aTrue = classifyAlt(src, targ)
            aPred = classifyAlt(src, pred)
            print("for", src, targ, "the alt pattern is", aTrue, "for", pred, "it is", aPred)
            
            if aPred in [ ("e", "ie"), #sentir ~ siento
                          ("o", "ue"), #mover ~ muevo
                          ("e", "i"),  #teñir ~ tiño
                      ]:
                outcome = "natural_alt"
            elif correct:
                outcome = "other_correct"
            elif aPred[0] != aPred[1]:
                if stressShift(aPred):
                    outcome = "stress"
                else:
                    outcome = "unnatural_alt"
            else:
                outcome = "no_alt"

            print("->", outcome)

            if srcRule == "abacorar" and exeRule == "acensuar":
                print("XXX", src, pred, aTrue, aPred, outcome)

            outcomes[src][outcome] += 1
            outcomesByEx[srcRule, exeRule][outcome] += 1

    for src, result in outcomes.items():
        print(src)
        for ri, ct in result.most_common():
            print("\t", ri, ct)
        print()

    sameOutcome = Counter()
    for (src, pred), exes in byTarget.items():
        for ex, ct in exes.items():
            for ex2, ct2 in exes.items():
                if ex == ex2:
                    continue

                key = (ex, ex2)
                sameOutcome[key] += min(ct, ct2)

    for key, ct in sameOutcome.most_common():
        print(key, "\t", ct)

    with open(of, "w") as ofh:
        writer = csv.DictWriter(ofh, ["src.rule", "ex.rule", "natural.alt", "other.correct", "stress", "unnatural.alt", "no.alt"])
        writer.writeheader()
        for (srcRule, exeRule), stats in outcomesByEx.items():
            writer.writerow({ "src.rule" : srcRule, "ex.rule" : exeRule,
                              "natural.alt" : stats["natural_alt"],
                              "other.correct" : stats["other_correct"],
                              "unnatural.alt" : stats["unnatural_alt"],
                              "stress" : stats["stress"],
                              "no.alt" : stats["no_alt"]})

    with open(of2, "w") as ofh:
        ofh.write("class1,class2,same.outcome\n")

        for key, ct in sameOutcome.most_common():
            ofh.write(",".join([key[0], key[1], str(ct)]) + "\n")
