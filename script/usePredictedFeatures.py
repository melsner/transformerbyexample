import os
import sys
from writeOutput import readPreds

def paired(fh):
    buff = []
    for it in readPreds(fh):
        buff.append(it)
        if len(buff) == 2:
            yield buff
            buff = []

def splitSource(source):
    ind = source.index("TRG_")
    feats = source[ind:]
    analogy = source[:ind].strip()

    feats = feats.split()
    feats = [xx.replace("TRG_", "") for xx in feats]
    return analogy, feats

if __name__ == "__main__":
    predFile = sys.argv[1]
    outFile = os.path.splitext(sys.argv[1])[0] + "-predicted.txt"

    inst = 0
    fAcc = 0
    output = []

    with open(predFile, encoding="utf-8") as fh:
        for ((src, trg, pred), (src, tfeat, pfeat)) in paired(fh):
            # print("SRC", src)
            # print("TRG", trg)
            # print("PRD", pred)
            # print("F1", tfeat, "F2", pfeat)
            # print()

            inst += 1
            if tfeat == pfeat:
                fAcc += 1

            source, sourceFeats = splitSource(src)
            feats = sourceFeats + pfeat.strip().split()
            feats = [xx for xx in feats if xx != "CLASSIFY"]
            feats = ";".join(feats)

            # print("SNOFEAT !%s!" % source)
            # print("SFEAT", sourceFeats)
            # print("PFEAT", pfeat)
            # print("FEAT", feats)
            # print()

            output.append("%s\t%s\t%s\n" % (source, trg, feats))

    print("Feature accuracy:", fAcc, "/", inst, "\t", fAcc / inst)

    with open(outFile, "w", encoding="utf-8") as ofh:
        for line in output:
            ofh.write(line)
