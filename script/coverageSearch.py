import sys
import os
import re
import csv
import numpy as np

from collections import defaultdict, Counter
from utils import readPreds, getEditClass
from outputsByClass import *

if __name__ == "__main__":
    run = sys.argv[1]
    searchLemma = sys.argv[2]
    searchForm = sys.argv[3]

    langFile = "../2021Task0/part2/eng.train"
    with open(langFile) as lang:
        for line in lang:
            (lemma, form, feats, orth1, orth2) = line.strip().split("\t")
            if orth1 == searchLemma and orth2 == searchForm:
                targLemma = lemma
                targForm = form

    targLemma = "".join(targLemma.split())
    targForm = "".join(targForm.split())

    print("IPA:", targLemma, targForm)

    targRule = getEditClass(targLemma, targForm)
    print("Edit rule:", targRule)

    asExe = 0
    asSrc = 0
    asBoth = 0
    srcs = set()

    td = run + "/train.txt"
    with open(td) as tfh:
        for line in tfh:
            (src, targ, feats) = line.strip().split("\t")
            srcLemma, rest = src.split(":")
            exeLemma, exeForm = rest.split(">")

            if targ.startswith("#FEATS"):
                continue
            #print(srcLemma, exeLemma, exeForm, targ, "!!")

            srcRule = getEditClass(srcLemma, targ)
            exeRule = getEditClass(exeLemma, exeForm)
            if srcRule == targRule:
                asSrc += 1
                srcs.add(srcLemma)
            if exeRule == targRule:
                asExe += 1
                srcs.add(exeLemma)
            if exeRule == targRule and srcRule == targRule:
                asBoth += 1

    print("Rule count as src/targ:", asSrc)
    print("Rule count as exemplar:", asExe)
    print("Rule count as both:", asBoth)
    print("Supporting words:", srcs)
