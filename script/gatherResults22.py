import os
import sys
import re
import shutil

def lgFind(lang, pth):
    for root, dirs, files in os.walk(pth):
        if lang in files:
            return True

origPath = sys.argv[1]
dpath = sys.argv[2]
opath = sys.argv[3]
for root, dirs, files in os.walk(dpath):
    for fi in files:
        if fi.endswith("predictions-std.txt"):
            expt = os.path.abspath(root + "/../")
            expt = os.path.basename(expt)
            if "-test" not in expt:
                language = expt.replace("-test", "")

                if lgFind(language + "_large.train", origPath):
                    size = "_large"
                elif lgFind(language + "_small.train", origPath):
                    size = "_small"
                else:
                    size = ""

                #ofile = opath + "/" + language + size + ".test"
                ofile = opath + "/" + language + size + ".dev"
                print(root + "/" + fi, "->", ofile)
                shutil.copy(root + "/" + fi, ofile)
