import sys
import os

if __name__ == "__main__":
    jobd = sys.argv[1]
    rund = sys.argv[2]

    jlist = []

    for fi in os.listdir(jobd):
        fn = fi.replace(".sh", ".err")
        if not os.path.exists(rund + "/" + fn):
            print(fi, end=", ")
            jlist.append(fi)
    print()

    for ji in jlist[:5]:
        cmd = "sbatch %s/%s" % (jobd, ji)
        print(cmd)
        os.system(cmd)
