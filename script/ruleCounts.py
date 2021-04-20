import sys
import pandas as pd
import re

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])

    df["pred"] = df["pred"].astype(str)

    print(df["appliedRule"].count())

    print(df.groupby("appliedRule").count().sort_values(by="targ", ascending=False))

    selected = df[df["correct"] == 1]
    print(selected)

    print(selected.groupby("appliedRule").count().sort_values(by="targ", ascending=False))
    print(len(selected.groupby("appliedRule").count().sort_values(by="targ", ascending=False)))


    #print(df.groupby("rule").count().sort_values(by="targ", ascending=False))
