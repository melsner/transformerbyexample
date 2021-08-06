from __future__ import division, print_function
import sys
import re
from collections import defaultdict
import os
import numpy as np
import argparse
import random
import math
import six
if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl

import networkx as nx

from Seq2seq import seq2seq_runner, dataloader
from Seq2seq import model as model_lib

from s2sFlags import *
from utils import edist_alt, edist, cacheWipe, get_size, findLatestModel
from classifyEditUtil import *

def loadGraphs(data, dd):
    res = {}
    for fi in os.listdir(dd):
        if fi.endswith(".pkl"):
            feats = fi.replace(".pkl", "").split("-")
            lang = feats.pop(-1)

            graph = pkl.load(open(dd + "/" + fi, "rb"))

            # print("Read", feats, lang)
            # print("Inventory of nodes", graph.nodes())

            symmGr = symmetrize(graph)
            distrs = graphToDistrs((data, feats, lang), symmGr)

            res[frozenset(feats), lang] = distrs

    print("Computed all graphs...")

    return res

def symmetrize(graph):
    symmGr = nx.Graph()
    for (n0, n1) in graph.edges():
        ej = graph[n1].get(n0, None)
        ei = graph[n0][n1]
        sdi = ei["loss"]
        
        if n0 == n1:
            continue

        if ej != None:
            sdj = ej["loss"]

            avg = (sdi + sdj) / 2
            symmGr.add_edge(n0, n1, weight=avg)
        else:
            symmGr.add_edge(n0, n1, weight=sdi)

    return symmGr

def graphToDistrs(data, graph):
    res = {}
    for ni in graph.nodes():
        #print("computations for", ni)
        #dist = nodeDistr(graph, ni)
        dist = nodeDistrByType(data, graph, ni)
        res[ni] = dist
    return res

def nodeDistr(graph, ni):
    dist = {}
    for nj in graph.nodes():
        if nj == ni:
            continue

        if nj in graph[ni]:
            edge = graph[ni][nj]
            weight = edge["weight"]
        else:
            weight = None

        if weight is None or weight > 6:
            weight = 0
        elif weight < 2:
            weight = .5
        else:
            weight = 2 / weight

        dist[nj] = weight

    total = sum(dist.values())
    #print("dist at this point", dist, "sum", total)

    if total > .5:
        dist = { key : .5 * (val / total) for (key, val) in dist.items() }
        total = sum(dist.values())

    dist[ni] = 1 - total

    # print("final dist")
    # for key, val in dist.items():
    #     if val > 0:
    #         print(key, "\t", val)
    return dist

def nodeDistrByType(data, graph, ni):
    (data, feats, lang) = data
    dist = {}
    for nj in graph.nodes():
        classID = data.internEC[frozenset(feats), lang].get(nj, None)
        if classID is None:
            nWords = 1
        else:
            words = data.byEditClass[frozenset(feats), lang].get(classID, [None,])
            nWords = len(words)

        #print("class", nj, "has", nWords, "words")

        if ni == nj:
            weight = 0
        elif nj in graph[ni]:
            edge = graph[ni][nj]
            weight = edge["weight"]
        else:
            weight = 100

        if weight < 6:
            dist[nj] = nWords

    total = sum(dist.values())

    dist = { key : (val / total) for (key, val) in dist.items() }

    if len(dist) > 1:
        print("Info for", "".join(ni), ":")
        for kk, vv in sorted(dist.items(), key=lambda xx: xx[1], reverse=True):
            if vv > 0:
                print("".join(kk), vv)
        print()

    return dist
