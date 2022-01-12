import sys
import os
import re
import csv
import numpy as np

from collections import defaultdict, Counter
from utils import readPreds, getEditClass
from outputsByClass import *

if __name__ == "__main__":
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

    if True:
        langFile = sys.argv[1]
        byRule = defaultdict(lambda: defaultdict(list))
        with open(langFile) as lang:
            for line in lang:
                (lemma, form, feats) = line.strip().split("\t")
                rule = getEditClass(lemma, form)

                if re.search("e[^aeiou]{1,3}ir$", lemma) and re.search("i[^aeiou]{1,3}[ie]n$", form):
                    print(lemma, form, rule)

                if rule not in names:
                    continue

                fs = frozenset(feats.split(";"))
                byRule[fs][rule].append((lemma, form, feats))
    
        out = sys.argv[2]
        with open(out, "w") as outfh:
            for fs in byRule:
                print(len(byRule[fs]), "classes for cell", fs, "of which", 
                      len([ri for (ri, mi) in byRule[fs].items() if len(mi) > 20]), "> 20")
                for rule, mems in sorted(byRule[fs].items(), key=lambda xx: len(xx[1]), reverse=True):
                    if len(mems) > 0:
                        print(" ".join(rule), mems[0], len(mems), names[rule])
                    
                        np.random.shuffle(mems)
                        samples = mems[:20]
                        for si in samples:
                            outfh.write("\t".join(si) + "\n")
    
                    else:
                        pass
                        #print("skip")
                
                print("\n\n")
