import sys
import os
from collections import defaultdict, Counter

from universalbackground import discoverVocab
from byexample import Data, createDataFromDir

from s2sFlags import *

def acceptableChars(charSets, families, threshold=.5):
    charStats = defaultdict(Counter)
    famCounts = Counter()

    for (lang, fam) in families.items():
        print("%s is part of the %s family" % (lang, fam))
        famCounts[fam] += 1

        cset = charSets[lang]
        for char in cset:
            charStats[fam][char] += 1

    acceptable = Counter()

    for fam, stats in charStats.items():
        for char, inFams in stats.items():
            prop = inFams / famCounts[fam]
            print("%s is in %f of %s languages" % (char, prop, fam))
            if prop > threshold:
                acceptable[char] += 1

    result = set()

    for ch, ct in acceptable.items():
        if ct == len(famCounts):
            print("Char", ch, "is acceptable")
            result.add(ch)

    return result

def expand(template, vowels, consonants):
    res = []
    for ch in template:
        if ch == "V":
            res.append(np.random.choice(list(vowels)))
        elif ch == "C":
            res.append(np.random.choice(list(consonants)))
        else:
            assert(0)

    return "".join(res)

def validityCheck(candidate, data, threshold=.01):
    hits = Counter()
    used = set()
    for (lemma, form, feats, lang, family) in data.instances:
        if (lemma, lang) in used:
            continue

        if form.startswith(candidate):
            hits[lang, "prefix"] += 1
            used.add((lemma, lang))
        elif form.endswith(candidate):
            hits[lang, "suffix"] += 1
            used.add((lemma, lang))

    #print("hits for", candidate, hits)
    for (lang, tp), ct in hits.items():
        prop = ct / len(data.langLemmas[lang])
        if prop > .001:
            print("Wonder if", candidate, "is a", tp, "in", lang, "affecting", prop, "lemmas")

            if prop > threshold:
                print("Too suspicious!")
                return False

    return True

def createAffix(template, vowels, consonants, data, threshold):
    while True:
        candidate = expand(template, vowels, consonants)
        if validityCheck(candidate, data, threshold):
            return candidate

def createLemma(template, vowels, consonants, data):
    while True:
        candidate = expand(template, vowels, consonants)
        if candidate not in data.allLemmas:
            return candidate

def getExemplar(lemmas, avoid):
    ex = avoid
    while ex == avoid:
        ex = np.random.choice(list(lemmas))
    return ex

def geminate(word, msounds, direction="last", nonFinal=False):
    if direction == "last":
        direc = -1
    else:
        direc = 1

    nonFinal = int(nonFinal) #ie, -1 if nonfinal mode is on

    for ind in np.arange(len(word) - nonFinal)[::direc]:
        if word[ind] in msounds:
            form = word[:ind] + word[ind] + word[ind:]
            return form

def shift(word, mapping, direction="last", nonFinal=False):
    if direction == "last":
        direc = -1
    else:
        direc = 1

    nonFinal = int(nonFinal) #ie, -1 if nonfinal mode is on

    for ind in np.arange(len(word) - nonFinal)[::direc]:
        if word[ind] in mapping:
            form = word[:ind] + mapping[word[ind]] + word[ind + 1:]
            return form

    return word

def makeGradations(lemmas, msounds, mode, nonFinal=False):
    mapping = {
        "a": "ei",
        "e": "i",
        "i": "ai",
        "o": "u",
        "u": "au",
        "b" : "p",
        "g" : "k",
        "d" : "t"
        }

    mapping = { kk : vv for (kk, vv) in mapping.items() if kk in msounds }
    for mi in msounds:
        if mi not in mapping:
            mapping[mi] = mi

    res = []
    for li in lemmas:
        ex = getExemplar(lemmas, li)

        if mode == "geminate":
            form = geminate(li, msounds, nonFinal=nonFinal)
            exForm = geminate(ex, msounds, nonFinal=nonFinal)

            res.append((li, form, ex, exForm))
        elif mode == "shift":
            form = shift(li, mapping, nonFinal=nonFinal)
            exForm = shift(ex, mapping, nonFinal=nonFinal)

            res.append((li, form, ex, exForm))            

    return res

def addAffix(word, affix, mode, vowels, epenthesize):
    if mode == "suffix":
        if epenthesize and word[-1] in vowels and affix[0] in vowels:
            form = word + epenthesize + affix
        else:
            form = word + affix
    elif mode == "prefix":
        if epenthesize and word[0] in vowels and affix[-1] in vowels:
            form = affix + epenthesize + word
        else:
            form = affix + word

    return form

def reduplicate(word, mode):
    if mode == "first":
        return word[:2] + word
    elif mode == "last":
        return word + word[-2:]

def makeReduplications(lemmas, mode):
    res = []
    for li in lemmas:
        ex = getExemplar(lemmas, li)
        form = reduplicate(li, mode)
        exForm = reduplicate(ex, mode)
        res.append((li, form, ex, exForm))

    return res

def makeAffixations(lemmas, affixes, mode, vowels=None, epenthesize=None):
    res = []
    for li in lemmas:
        if mode is None:
            ex = getExemplar(lemmas, li)
            res.append((li, li, ex, ex))

        else:
            for ai in affixes:
                ex = getExemplar(lemmas, li)
                form = addAffix(li, ai, mode, vowels, epenthesize)
                exForm = addAffix(ex, ai, mode, vowels, epenthesize)
                res.append((li, form, ex, exForm))

    return res

def writeInsts(insts, fh, settings):
    for (lemma, form, exLemma, exForm) in insts:
        src = "%s:%s>%s" % (lemma, exLemma, exForm)
        for feats in settings:
            featStr = ";".join(feats)
            fh.write("%s\t%s\t%s\n" % (src, form, featStr))

if __name__ == "__main__":
    args = get_arguments()

    run = args.run
    dpath = args.data
    mode = args.synthetic_dev_type
    alphabet = args.synthetic_dev_alphabet


    if alphabet == "common":
        charSets, families, feats, maxLemma, maxForm, lemmaLengths, formLengths = discoverVocab(dpath)

        cset = acceptableChars(charSets, families)
    elif alphabet == "cyrillic":
        cset = set("абвгдеёжзийклмнопрстуфхцчшщэюя")
    elif alphabet == "target":
        cset = set(charSets["deu"].keys())
        for lang in ["deu", "tgl", "ote", "swa", "fin"]:
            cs = charSets[lang]
            cset = cset2.intersection(set(cs.keys()))
    else:
        assert(0)

    vowels = cset.intersection("aeiouаеёоуэюя") #fine for the set we actually get which contains no umlauts
    consonants = cset.difference(vowels)

    data = createDataFromDir(dpath, args)

    data.langLemmas = defaultdict(set)
    data.allLemmas = set()
    for (lemma, form, feats, lang, family) in data.instances:
        data.langLemmas[lang].add(lemma)
        data.allLemmas.add(lemma)

    os.system("rm -rf %s" % run)
    os.makedirs(run)

    if mode == "affix":
        affixes = set()
        while len(affixes) < 10:
            affix = createAffix("VCV", vowels, consonants, data, .01)
            #print("Accepted affix", affix)
            affixes.add(affix)

        while len(affixes) < 20:
            affix = createAffix("CV", vowels, consonants, data, .05)
            #print("Accepted affix", affix)
            affixes.add(affix)

        while len(affixes) < 30:
            affix = createAffix("CVCV", vowels, consonants, data, .01)
            #print("Accepted affix", affix)
            affixes.add(affix)

        affixes.add("s")
        affixes.add("k")
        affixes.add("r")
        affixes.add("m")
        affixes.add("a")
        affixes.add("e")
        affixes.add("i")

        print("Affixes:", sorted(affixes, key=lambda xx: len(xx)))
        print()

    lemmas = set()
    while len(lemmas) < 30:
        lemma = createLemma("CVCV", vowels, consonants, data)
        #print("Accepted lemma", lemma)
        lemmas.add(lemma)

    while len(lemmas) < 60:
        lemma = createLemma("CVCVC", vowels, consonants, data)
        #print("Accepted lemma", lemma)
        lemmas.add(lemma)

    while len(lemmas) < 90:
        lemma = createLemma("CVCVCVC", vowels, consonants, data)
        #print("Accepted lemma", lemma)
        lemmas.add(lemma)

    print("Lemmas:", sorted(lemmas, key=lambda xx: len(xx)))
    print()

    if mode == "reduplicate":
        insts = makeReduplications(lemmas, "first")
        ofile = "%s/reduplicate-first.txt" % run
        with open(ofile, "w") as ofh:
            writeInsts(insts, ofh, [
                ["FAM_austronesian",], ["LANG_tgl", "FAM_austronesian"],
                ["FAM_germanic",], ["LANG_deu", "FAM_germanic"],
                ["FAM_uralic",], ["LANG_fin", "FAM_uralic"],
                ["FAM_niger-congo",], ["LANG_swa", "FAM_niger-congo"],
                ["FAM_oto-manguean",], ["LANG_ote", "FAM_oto-manguean"],
                ["FAM_synthetic",], ["LANG_suffix", "FAM_synthetic"],
            ])

        insts = makeReduplications(lemmas, "last")
        ofile = "%s/reduplicate-last.txt" % run
        with open(ofile, "w") as ofh:
            writeInsts(insts, ofh, [
                ["FAM_austronesian",], ["LANG_tgl", "FAM_austronesian"],
                ["FAM_germanic",], ["LANG_deu", "FAM_germanic"],
                ["FAM_uralic",], ["LANG_fin", "FAM_uralic"],
                ["FAM_niger-congo",], ["LANG_swa", "FAM_niger-congo"],
                ["FAM_oto-manguean",], ["LANG_ote", "FAM_oto-manguean"],
                ["FAM_synthetic",], ["LANG_suffix", "FAM_synthetic"],
            ])

    if mode == "grade":
        insts = makeGradations(lemmas, vowels, "geminate")
        ofile = "%s/geminateV.txt" % run
        with open(ofile, "w") as ofh:
            writeInsts(insts, ofh, [
                ["FAM_austronesian",], ["LANG_tgl", "FAM_austronesian"],
                ["FAM_germanic",], ["LANG_deu", "FAM_germanic"],
                ["FAM_uralic",], ["LANG_fin", "FAM_uralic"],
                ["FAM_niger-congo",], ["LANG_swa", "FAM_niger-congo"],
                ["FAM_oto-manguean",], ["LANG_ote", "FAM_oto-manguean"],
                ["FAM_synthetic",], ["LANG_suffix", "FAM_synthetic"],
            ])

        insts = makeGradations(lemmas, consonants, "geminate", nonFinal=True)
        ofile = "%s/geminateC.txt" % run
        with open(ofile, "w") as ofh:
            writeInsts(insts, ofh, [
                ["FAM_austronesian",], ["LANG_tgl", "FAM_austronesian"],
                ["FAM_germanic",], ["LANG_deu", "FAM_germanic"],
                ["FAM_uralic",], ["LANG_fin", "FAM_uralic"],
                ["FAM_niger-congo",], ["LANG_swa", "FAM_niger-congo"],
                ["FAM_oto-manguean",], ["LANG_ote", "FAM_oto-manguean"],
                ["FAM_synthetic",], ["LANG_suffix", "FAM_synthetic"],
            ])

        insts = makeGradations(lemmas, vowels, "shift")
        ofile = "%s/shiftV.txt" % run
        with open(ofile, "w") as ofh:
            writeInsts(insts, ofh, [
                ["FAM_austronesian",], ["LANG_tgl", "FAM_austronesian"],
                ["FAM_germanic",], ["LANG_deu", "FAM_germanic"],
                ["FAM_uralic",], ["LANG_fin", "FAM_uralic"],
                ["FAM_niger-congo",], ["LANG_swa", "FAM_niger-congo"],
                ["FAM_oto-manguean",], ["LANG_ote", "FAM_oto-manguean"],
                ["FAM_synthetic",], ["LANG_suffix", "FAM_synthetic"],
            ])

        insts = makeGradations(lemmas, consonants, "shift")
        ofile = "%s/shiftC.txt" % run
        with open(ofile, "w") as ofh:
            writeInsts(insts, ofh, [
                ["FAM_austronesian",], ["LANG_tgl", "FAM_austronesian"],
                ["FAM_germanic",], ["LANG_deu", "FAM_germanic"],
                ["FAM_uralic",], ["LANG_fin", "FAM_uralic"],
                ["FAM_niger-congo",], ["LANG_swa", "FAM_niger-congo"],
                ["FAM_oto-manguean",], ["LANG_ote", "FAM_oto-manguean"],
                ["FAM_synthetic",], ["LANG_suffix", "FAM_synthetic"],
            ])

    if mode == "affix":
        insts = makeAffixations(lemmas, affixes, "suffix")
        ofile = "%s/suffixes.txt" % run
        with open(ofile, "w") as ofh:
            writeInsts(insts, ofh, [
                ["FAM_austronesian",], ["LANG_tgl", "FAM_austronesian"],
                ["FAM_germanic",], ["LANG_deu", "FAM_germanic"],
                ["FAM_uralic",], ["LANG_fin", "FAM_uralic"],
                ["FAM_niger-congo",], ["LANG_swa", "FAM_niger-congo"],
                ["FAM_oto-manguean",], ["LANG_ote", "FAM_oto-manguean"],
                ["FAM_synthetic",], ["LANG_suffix", "FAM_synthetic"],
            ])

        insts = makeAffixations(lemmas, affixes, "prefix")
        ofile = "%s/prefixes.txt" % run
        with open(ofile, "w") as ofh:
            writeInsts(insts, ofh, [
                ["FAM_austronesian",], ["LANG_tgl", "FAM_austronesian"],
                ["FAM_germanic",], ["LANG_deu", "FAM_germanic"],
                ["FAM_uralic",], ["LANG_fin", "FAM_uralic"],
                ["FAM_niger-congo",], ["LANG_swa", "FAM_niger-congo"],
                ["FAM_oto-manguean",], ["LANG_ote", "FAM_oto-manguean"],
                ["FAM_synthetic",], ["LANG_suffix", "FAM_synthetic"],
            ])

        insts = makeAffixations(lemmas, affixes, None)
        ofile = "%s/copy.txt" % run
        with open(ofile, "w") as ofh:
            writeInsts(insts, ofh, [
                ["FAM_austronesian",], ["LANG_tgl", "FAM_austronesian"],
                ["FAM_germanic",], ["LANG_deu", "FAM_germanic"],
                ["FAM_uralic",], ["LANG_fin", "FAM_uralic"],
                ["FAM_niger-congo",], ["LANG_swa", "FAM_niger-congo"],
                ["FAM_oto-manguean",], ["LANG_ote", "FAM_oto-manguean"],
                ["FAM_synthetic",], ["LANG_suffix", "FAM_synthetic"],
            ])
