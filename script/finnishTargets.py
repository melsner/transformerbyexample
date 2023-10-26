import sys
import os
import re
import csv
import numpy as np

from collections import defaultdict, Counter
from utils import readPreds, getEditClass
from outputsByClass import *

if __name__ == "__main__":

    if True:
        langFile = sys.argv[1]
        byRule = defaultdict(lambda: defaultdict(list))
        with open(langFile) as lang:
            for ind, line in enumerate(lang):
                (lemma, form, feats) = line.strip().split("\t")
                fs = frozenset(feats.split(";"))

                #Finnish
                #if fs != set(["N", "IN+ESS", "PL"]):
                #Icelandic
                #if fs != set(["V", "NOM(1,SG)", "IND", "PRS"]):
                #irish
                #if fs != set(["N", "GEN(SG)"]):
                #khaling
                #if fs != set(["V", "PRS", "1", "SG", "INTR", "POS"]):
                #nth sami
                if fs != set(["N", "COM", "SG"]):
                    continue

                rule = getEditClass(lemma, form)

                byRule[fs][rule].append((lemma, form, feats))
            
            if ind % 5000 == 0:
                print(len(byRule), sorted([len(xx) for xx in byRule]))
    
    for fs in byRule:
        print(len(byRule[fs]), "classes for cell", fs, "of which", 
              len([ri for (ri, mi) in byRule[fs].items() if len(mi) > 5]), "> 5")
        for rule, mems in sorted(byRule[fs].items(), key=lambda xx: len(xx[1]), reverse=True):
            if len(mems) > 5:
                print(" ".join(rule), mems[0], len(mems))
        print()

    out = sys.argv[2]
    with open(out, "w") as outfh:
        for fs in byRule:
            for rule, mems in sorted(byRule[fs].items(), key=lambda xx: len(xx[1]), reverse=True):
                if len(mems) > 5:
                    np.random.shuffle(mems)
                    samples = mems[:20]
                    for si in samples:
                        outfh.write("\t".join(si) + "\n")
