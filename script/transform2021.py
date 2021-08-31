import sys
import os

if __name__ == "__main__":
    dpath = sys.argv[1]
    opath = sys.argv[2]

    os.makedirs(opath, exist_ok=True)
    os.makedirs(opath + "/germanic", exist_ok=True)
    #where is Russian?

    for root, dirs, files in os.walk(dpath):
        for fi in files:
            if fi.endswith(".train"):
                target = opath + "/germanic/" + os.path.splitext(fi)[0] + ".trn"
                with open(target, "w") as ofh:
                    for line in open(root + "/" + fi):
                        (lemma, form, feats, orth1, orth2) = line.strip().split("\t")
                        lemma = lemma.replace(" ", "")
                        form = form.replace(" ", "")
                        ofh.write("%s\t%s\t%s\n" % (lemma, form, feats))
