import sys
import pandas as pd
import re
from utils import edist_alt

def chunkRule(rule):
    currType = None
    seq = []
    curr = []
    for chunk in rule:
        chunkType = chunk[0]
        if chunkType == currType:
            curr.append(chunk)
        else:
            curr = []
            currType = chunk[0]
            seq.append(curr)
            curr.append(chunk)

    if not seq[-1]:
        seq = seq[:-1]

    return seq

def locateChunks(rule):
    seq = chunkRule(rule)

    stemInds = []
    for ind, subseq in enumerate(seq):
        if subseq[0] == "*":
            stemInds.append(ind)

    res = []
    for ind, subseq in enumerate(seq):
        if ind in stemInds:
            stemBefore = None
            stemAfter = None
        else:
            stemBefore = stemInds and ind > stemInds[0]
            stemAfter = stemInds and ind < stemInds[-1]

        for char in subseq:
            res.append((char, ind, stemBefore, stemAfter))

    return res

def ruleDescription(rule):
    res = set()

    for chunk, chunkInd, stemBefore, stemAfter in locateChunks(rule):
        if chunk == "*":
            continue
        elif not stemBefore:
            res.add("TRG_RULE_PREF")
        elif not stemAfter:
            res.add("TRG_RULE_SUFF")
        elif stemBefore and stemAfter:
            res.add("TRG_RULE_STEM")
        else:
            #pathological: rule with no identifiable stem content
            res.add("TRG_RULE_SUPPLETIVE")

    return res

def ruleContrast(exRule, targRule):
    cost, (alt1, alt2) = edist_alt(exRule, targRule)
    #print("aligned", exRule, targRule)
    #print(alt1, alt2)
    chunks = locateChunks(targRule)

    #print("CH", chunks)

    #restructure into chunked sublists
    seq = []
    currId = -1
    curr = []
    for (item, status), (_, chunkId, stemBefore, stemAfter) in zip(alt2, chunks):
        if chunkId != currId:
            curr = []
            seq.append(curr)
            currId = chunkId

        curr.append((item, status, stemBefore, stemAfter))

    #print("AGG:", seq)

    res = set()
    for subseq in seq:
        #print("\t", subseq)
        stem = (subseq[0][0] == "*")
        if stem:
            assert(len(subseq) == 1)
            continue

        stemBefore = subseq[0][2]
        stemAfter = subseq[0][3]
        statuses = [xx[1] for xx in subseq]
        if stemBefore and not stemAfter:
            affixType = "SUFF"
        elif stemAfter and not stemBefore:
            affixType = "PREF"
        else:
            affixType = "IN"

        if subseq[0][0].startswith("-"):
            affixType += "_RM"

        if all(statuses):
            cat = "TRG_CMP_%s_COPY" % affixType
        elif sum(statuses) / len(statuses) < .5:
            cat = "TRG_CMP_%s_REPLACE" % affixType
        else:
            statusAltDirs = statusAlterations(statuses)
            if statusAltDirs == (stemBefore, False, stemAfter):
                cat = "TRG_CMP_%s_CONTACT" % affixType
            else:
                cat = "TRG_CMP_%s_DISTANT" % affixType

        #print("\t", cat)
        res.add(cat)

    return res

def statusAlterations(statuses):
    left = not statuses[0]
    right = not statuses[-1]
    middle = False
    stemL = False
    gap = False
    for xi in statuses:
        if not xi and stemL:
            gap = True
        if xi:
            stemL = True
            if gap:
                middle = True

    return (left, middle, right)

def featureFn(lang, fam, cell, targRule, exRule, extraFeatures=False):
    features = ["TRG_LANG_%s" % lang, "TRG_FAM_%s" % fam]

    if extraFeatures:
        features += ["TRG_CELL_%s" % xx for xx in cell]
        features += list(ruleDescription(targRule))
        features += list(ruleContrast(exRule, targRule))

    return features

def classificationInst(src, targ, features):
    #the "src" part is the source lemma and the exemplar
    #split out the language stuff from the other features
    srcFeats = [xx for xx in features if "LANG_" in xx or "FAM_" in xx]
    srcFeats = ["TRG_CLASSIFY"] + srcFeats
    targFeats = [xx for xx in features if "LANG_" not in xx and "FAM_" not in xx]
    sortedTargFeats = sorted(targFeats)
    targFStr = "#FEATS;" + ";".join(sortedTargFeats)
    output = "%s\t%s\t%s\n" % (src, targFStr, srcFeats)

    return output

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    rules = list(df["appliedRule"])
    rules = [eval(xx) for xx in rules]
    print(rules[:5])

    for ri in rules[:5]:
        print(ri, locateChunks(ri))

    print("------------------------\n\n")

    for ri in rules[:5]:
        print(ri, ruleDescription(ri))

    print("------------------------\n\n")

    for ri in rules[1:5]:
        for rj in rules[14:15]:
            print(ri, rj, ruleContrast(ri, rj))
