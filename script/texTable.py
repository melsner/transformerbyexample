import sys

for line in sys.stdin:
    flds = line.strip().split("\t")
    print(" & ".join(flds) + "\\\\")
