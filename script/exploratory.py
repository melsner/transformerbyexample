import sys
import pandas as pd
import re

def hasReduplicant(row):
    targ = row.targ
    reduplicant = re.search("(..)\\1", targ)
    if not reduplicant:
        return 0
    reduplicant = reduplicant.group(1)
    reduplicant += reduplicant
    #print(targ, "red", reduplicant, "pred", row.pred, reduplicant in row.pred)
    if reduplicant in row.pred:
        return 1
    return 0

def hasGeminate(row):
    pred = row.pred
    if re.search("(.)\\1", pred):
        #print("Geminate detected", row.lemma, row.pred, row.lang, row.fam, "int.", row.targ)
        return 1
    return 0

def ruleSeg(row, suffix=True):
    affix = row.affix
    lemma = row.lemma
    pred = row.pred
    if suffix:
        intended = lemma + affix
    else:
        intended = affix + lemma
    if affix in pred:
        return 1
    else:
        return 0

def ruleCatCV(rule):
    return ruleCat(rule, cv=True)

def ruleCat(rule, cv=False):
    added = 0
    deleted = 0

    addedV = 0
    deletedV = 0

    for ri in rule:
        if ri.startswith("+"):
            if ri[1] in "aeiou":
                deletedV += 1
            deleted += 1
        elif ri.startswith("-") and len(ri) == 2:
            added += 1
            if ri[1] in "aeiou":
                addedV += 1

    if deleted >= 4 or added >= 4:
        res = "X"
    else:
        if cv:
            addedC = (added - addedV)
            deletedC = (deleted - deletedV)

            addedC = min(addedC, 2)
            addedV = min(addedV, 2)
            deletedC = min(deletedC, 2)
            deletedV = min(deletedV, 2)

            res = ""
            if addedV > 0:
                res += "+%dV" % addedV
            if addedC > 0:
                res += "+%dC" % addedC
            if deletedV > 0:
                res += "-%dV" % deletedV
            if deletedC > 0:
                res += "-%dC" % deletedC

        else:
            res = "-%d+%d" % (deleted, added)

    #print("categorized rule", rule, "as", res)
    return res

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])

    suffix = False
    prefix = False
    copy = False
    if "-suffixes.txt" in sys.argv[1]:
        print("Suffixes")
        suffix = True
    elif "-prefixes.txt" in sys.argv[1]:
        print("Prefixes")
        prefix = True
    elif "-copy.txt" in sys.argv[1]:
        print("Copy")
        copy = True

    df["pred"] = df["pred"].astype(str)

    #accuracy by language/family
    df["lang"] = df["lang"].astype(str)
    df["lang"] = df["lang"].replace("nan", "")
    df["setting"] = df.fam + "/" + df.lang
    print("Mean correct:", df["correct"].mean())

    print(df.groupby(["setting"]).mean())
    #sys.exit(1)

    # corr = df.groupby(["setting"]).mean()
    # def nanfirst(ser):
    #     return ser.replace("/nan", "")
    # print(corr["setting"].replace("/nan", ""))
    # corr = corr.sort_values(by="setting", key=nanfirst)
    # print(corr)

    print()

    df.rule = df.rule.map(eval)
    df["ruletype"] = df.rule.map(ruleCat)
    df["ruletypeCV"] = df.rule.map(ruleCatCV)
    df["ruleseg"] = df.apply(lambda xx: ruleSeg(xx, suffix=suffix), axis=1)
    df["ignored"] = (df.lemma == df.pred)
    df["geminated"] = df.apply(hasGeminate, axis=1)
    df["reduplicated"] = df.apply(hasReduplicant, axis=1)

    df["exCV1"] = df.exemplar.str[0:2]
    df["prefix.for.redup"] = df.exCV1 + df.lemma
    df["infix.for.redup"] = df.lemma.str[0:2] + df.exCV1 + df.lemma.str[2:]

    df["redup.out.type"] = "other"
    df.loc[df.pred == df.targ, "redup.out.type"] = "reduplicated"
    df.loc[df.pred == df["prefix.for.redup"], "redup.out.type"] = "prefixed"
    df.loc[df.pred == df["infix.for.redup"], "redup.out.type"] = "infixed"
    df.loc[df.pred == df["lemma"], "redup.out.type"] = "unaltered"

    if suffix:
        df["interface"] = df.lemmaCV.str[-1] + df.affixCV.str[0]
        df["lemma/affix"] = df.lemmaCV + "/" + df.affixCV
    elif prefix:
        df["interface"] = df.affixCV.str[-1] + df.lemmaCV.str[0] 
        df["lemma/affix"] = df.affixCV + "/" + df.lemmaCV
    elif copy:
        df["interface"] = ""
        df["lemma/affix"] = df.lemmaCV

    dfFam = df[df.lang == "nan"]

    print("Accuracy by cv")
    if prefix or suffix:
        feature = "affixCV"
    else:
        feature = "lemmaCV"

    grouped = df.groupby(["setting", feature]).mean()

    pivot = pd.pivot_table(grouped, values="correct", columns=feature, index="setting")
    print(pivot)

    print("------------")

    if prefix or suffix:
        print("Ruleseg: test if lemma+affix in output")
        grouped = df.groupby(["setting", feature]).mean()
        pivot = pd.pivot_table(grouped, values="ruleseg", columns=feature, index="setting")
        print(pivot)

    print("------------")

    print("Note to self: -?? means characters are deleted from target, +?? means characters are added to prediction")
    print("Type of edit rule")
    grouped = dfFam.groupby(["setting", "ruletype"]).count()
    pivot = pd.pivot_table(grouped, values="lemma", columns="ruletype", index="setting").fillna(0)
    #pd.options.display.max_columns = None
    pivot = pivot.div(pivot.sum(axis=1), axis=0).round(2)
    print(pivot)

    print()
    print("----------------")
    print()

    grouped = df.groupby(["setting", "ruletype"]).count()
    pivot = pd.pivot_table(grouped, values="lemma", columns="ruletype", index="setting").fillna(0)
    #pd.options.display.max_columns = None
    pivot = pivot.div(pivot.sum(axis=1), axis=0).round(2)
    print(pivot)


    print("Ignored?")
    grouped = df.groupby(["setting", feature]).mean()
    pivot = pd.pivot_table(grouped, values="ignored", columns=feature, index="setting")
    print(pivot)


    print()
    print("----------------")
    print()

    print("Reduplication status")

    dfilt = df[df.affix != df.exCV1]

    ds1 = dfilt.loc[(dfilt.setting == "germanic/deu") & (dfilt["redup.out.type"] == "prefixed")]
    print(ds1)

    grouped = dfilt.groupby(["setting", "redup.out.type"]).count()
    pivot = pd.pivot_table(grouped, values="correct", columns="redup.out.type", index="setting").fillna(0)
    print(100 * pivot.div(pivot.sum(axis=1), axis=0))


    print()
    print("----------------")
    print()
