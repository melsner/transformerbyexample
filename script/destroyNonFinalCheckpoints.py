import os
import sys

if __name__ == "__main__":
    dpath = sys.argv[1]
    for root, dirs, files in os.walk(dpath):
        if os.path.basename(root) == "checkpoints":
            saves = []
            for fi in files:
                if fi.startswith("manual_save"):
                    saveN = int(fi.replace(".h5", "").replace("manual_save", ""))
                    saves.append((fi, saveN))
            saves.sort(key=lambda xx: xx[1], reverse=True)

            if saves:
                print("Dir:", root)
                print("Keeping", saves[0])
                print("Destroying", saves[1:])
                for si, sn in saves[1:]:
                    spath = root + "/" + si
                    os.unlink(spath)
