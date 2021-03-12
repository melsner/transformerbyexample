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

if __name__ == "__main__":
    devfile = sys.argv[1]
    srcChars = set()
    targChars = set()
    with open(devfile) as dfh:
        for line in dfh:
            (src, targ, dummy) = line.strip().split("\t")
            srcChars.update(src)
            targChars.update(targ)

    vocab = sys.argv[2]
    srcVoc = set()
    targVoc = set()
    with open(vocab) as vfh:
        for line in vfh:
            flds = line.strip().split("\t")
            if flds[0] == "src_vocab":
                srcVoc.add(flds[2])
            elif flds[0] == "vocab":
                srcVoc.add(flds[1])
                targVoc.add(flds[1])
            elif flds[0] == "trg_vocab":
                targVoc.add(flds[2])

    print("Data file:", len(srcChars), "source chars", len(targChars), "target chars")
    print("Vocab file:", len(srcVoc), "source chars", len(targVoc), "target chars")

    print("Chars in data but not vocab (source):")
    print("\t".join(srcChars.difference(srcVoc)))
    print("Chars in data but not vocab (target):")
    print("\t".join(targChars.difference(targVoc)))

    print()

    print("Chars in vocab but not data (source):")
    print("\t".join(srcVoc.difference(srcChars)))
    print("Chars in vocab but not data (target):")
    print("\t".join(targVoc.difference(targChars)))
