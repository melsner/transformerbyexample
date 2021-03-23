import sys

nf = 100000

ctr = -1
with open(sys.argv[1]) as fh:
    for ind, line in enumerate(fh):
        if ind % nf == 0:
            ctr += 1
            ofh = open(sys.argv[1] + str(ctr), "w")

        ofh.write(line)
