import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def leaveOneOut(design, output):
    preds = []
    nn = design.shape[0]
    for row in range(nn):
        indices = np.array([xx for xx in range(nn) if xx != row])
        ndesign = design[indices, :]
        nout = output[indices]
        regress = LinearRegression().fit(design, output)
        pred = regress.predict(design[row:row+1,:])
        preds.append(pred)
    return preds

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])

    lemma = False
    base = False

    factors = ["pct.top.10", "pr0", "supporting.words", "med.pr", "most.freq.support0", "found"]
    #factors = ["found"]
    #factors = ["pct.top.10", "pr0", "supporting.words", "med.pr", "most.freq.support0", "found"]
    design = df[factors]

    design = np.array(design)
    for col, name in enumerate(factors):
        if name not in ["pct.top.10", "found"]:
            design[:, col] = np.log(1+design[:, col])

    #add interactions
    for c1 in ["pct.top.10", "pr0"]:
        for c2 in ["supporting.words", "most.freq.support0", "found", "pr0", "med.pr"]:
            if c1 != c2:
                if c1 in factors and c2 in factors:
                    col1 = factors.index(c1)
                    col2 = factors.index(c2)
                    design = np.hstack([design, design[:, col1:col1+1] * design[:, col2:col2+1]])
                    factors += [factors[col1] + "*" + factors[col2]]

    if lemma:
        lemmas = df["lemma"].astype("category").cat.codes
        lemmaV = np.zeros((lemmas.shape[0], np.max(lemmas) + 1))
        lemmaV[np.arange(lemmas.shape[0]), lemmas] = 1
        design = np.hstack([design, lemmaV])

    print("Shape of design matrix", design.shape)

    output = df["centered.rating"]
    if lemma:
        output = df["rating"]
    output = np.array(output)
    print("Shape of output", output.shape)

    regress = LinearRegression().fit(design, output)
    print("R2", regress.score(design, output))
    print(list(zip(factors, regress.coef_)))
    preds = regress.predict(design)
    df["prediction"] = preds

    preds = leaveOneOut(design, output)
    df["prediction"] = preds

    if base:
        print("***BASE***")
        basePreds = []
        for ind, row in df.iterrows():
            if row["form"].endswith("d"):
                basePreds.append(1)
            else:
                basePreds.append(0)
        df["prediction"] = basePreds
        preds = basePreds

    print("R2 (l1o):", r2_score(output, preds))

    goodPairs = 0
    badPairs = 0
    for lemma in set(df["lemma"]):
        ls = df.loc[df["lemma"] == lemma,]
        ratings = []
        preds = []
        stars = ["", ""]
        for ind, row in ls.iterrows():
            ratings.append(row["rating"])
            preds.append(row["prediction"])

        stars[np.argmax(ratings)] = "*"

        if ((ratings[0] > ratings[1] and preds[0] > preds[1]) or
            (ratings[1] > ratings[0] and preds[1] > preds[0])):
            goodPairs += 1
        else:
            badPairs += 1
            print("Discordant pair:", lemma, "~", " ".join(
                ["%s%s" % (star, fi) for (star, fi) in zip(stars, ls["form"])]))

    print("Concordant pairs:", goodPairs, goodPairs / (badPairs + goodPairs))
    print("Discordant pairs:", badPairs, badPairs / (badPairs + goodPairs))

    original = sys.argv[2]
    out = open(sys.argv[1] + ".predictions", "w")
    for ii, line in enumerate(open(original)):
        (lemma, form, feats, rating) = line.strip().split("\t")
        out.write("%s\t%s\t%s\t%f\n" % (lemma, form, feats, df["prediction"][ii]))
