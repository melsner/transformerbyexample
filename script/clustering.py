from __future__ import division, print_function
import numpy as np

class Clustering:
    # { LEMMA : [array, { WF : ROW# } ]
    # self.lem_2_responsibilities

    # { LEMMA : { ROW# : WF } }
    # self.lem_2_row_2_wf

    # { WF : { CL# : True } }
    # self.wf_2_cluster

    # { CL# : { WF : True } }
    # self.cluster_2_wf

    # { CL# : { LEMMA : { WF : True } } }
    # self.cluster_2_lem_2_wf

    # { LEMMA : { WF : { CL# : bool } }
    # self.lem_2_wf_2_cluster

    def __init__(self, lem_2_wf, kCells):
        self.lem_2_wf = lem_2_wf
        self.kCells = kCells

        self.wf_2_cluster = {}
        self.cluster_2_wf = {}
        self.cluster_2_lem_2_wf = {}
        self.lem_2_wf_2_cluster = {}

    def clear(self):
        self.wf_2_cluster = {}
        self.cluster_2_wf = {}
        self.cluster_2_lem_2_wf = {}
        self.lem_2_wf_2_cluster = {}

    def syncretisms(self):
        res = 0
        for wf, assignedTo in self.wf_2_cluster.items():
            if len(assignedTo) > 1:
                res += 1
        return res

    def overabundances(self):
        res = 0
        for cl, lem_2_wf in self.cluster_2_lem_2_wf.items():
            if len(lem_2_wf) > 1:
                res += 1
        return res

    def nClustersUsed(self):
        for ci, sub in self.cluster_2_wf.items():
            assert(len(sub) > 0)
        return len(self.cluster_2_wf)

    def singleCluster(self):
        for lemma, sub in self.lem_2_wf.items():
            for wf in sub:
                #note: this design WILL NOT work for wfs with multiple lemmas
                self.moveToCluster(lemma, wf, 0)

    def addToCluster(self, lemma, wf, cluster):
        assert(0 <= cluster < self.kCells)
        if wf not in self.wf_2_cluster:
            self.wf_2_cluster[wf] = {}
        self.wf_2_cluster[wf][cluster] = True

        if cluster not in self.cluster_2_wf:
            self.cluster_2_wf[cluster] = {}
        self.cluster_2_wf[cluster][wf] = True

        if cluster not in self.cluster_2_lem_2_wf:
            self.cluster_2_lem_2_wf[cluster] = {}
        if lemma not in self.cluster_2_lem_2_wf[cluster]:
            self.cluster_2_lem_2_wf[cluster][lemma] = {}
        self.cluster_2_lem_2_wf[cluster][lemma][wf] = True

        if lemma not in self.lem_2_wf_2_cluster:
            self.lem_2_wf_2_cluster[lemma] = {}
        if wf not in self.lem_2_wf_2_cluster[lemma]:
            self.lem_2_wf_2_cluster[lemma][wf] = {}
        self.lem_2_wf_2_cluster[lemma][wf][cluster] = True

    def removeFromCluster(self, lemma, wf, cluster):
        del self.wf_2_cluster[wf][cluster]
        if len(self.wf_2_cluster[wf]) == 0:
            del self.wf_2_cluster[wf]

        del self.cluster_2_wf[cluster][wf]
        if len(self.cluster_2_wf[cluster]) == 0:
            del self.cluster_2_wf[cluster]

        del self.cluster_2_lem_2_wf[cluster][lemma][wf]
        if len(self.cluster_2_lem_2_wf[cluster][lemma]) == 0:
            del self.cluster_2_lem_2_wf[cluster][lemma]
        if len(self.cluster_2_lem_2_wf[cluster]) == 0:
            del self.cluster_2_lem_2_wf[cluster]

        del self.lem_2_wf_2_cluster[lemma][wf][cluster]
        if len(self.lem_2_wf_2_cluster[lemma][wf]) == 0:
            del self.lem_2_wf_2_cluster[lemma][wf]
        if len(self.lem_2_wf_2_cluster[lemma]) == 0:
            del self.lem_2_wf_2_cluster[lemma]

    def currCluster(self, lemma, wf):
        return wf_2_cluster.get(wf, None)

    def moveToCluster(self, lemma, wf, cluster):
        """Add form WF as form of LEMMA to CLUSTER, removing it from any other cluster."""
        oldCluster = self.currCluster(lemma, wf)

        if oldCluster is not None:
            self.removeFromCluster(lemma, wf, oldCluster)
        self.addToCluster(self, lemma, wf, cluster)

class ResponsibilityMatrices:
    def __init__(self, lem_2_wf, kCells):
        self.lem_2_wf = lem_2_wf
        self.kCells = kCells
        self.matrices = {}
        self.lem_2_row_2_wf = {}
        self.lem_2_wf_2_row = {}
        for lem, sub in self.lem_2_wf.items():
            nWfs = len(sub)
            self.matrices[lem] = np.zeros( (nWfs, self.kCells) )

            self.lem_2_row_2_wf[lem] = {}
            self.lem_2_wf_2_row[lem] = {}

            for row, wf in enumerate(sub):
                self.lem_2_row_2_wf[lem][row] = wf
                self.lem_2_wf_2_row[lem][wf] = row

    def assign(self, lemma, wf, cluster, value):
        row = self.lem_2_wf_2_row[lemma][wf]
        self.matrices[lemma][row][cluster] = value

    def transform(self, lemma, fn):
        self.matrices[lemma] = fn(self.matrices[lemma])

    def transformAll(self, fn):
        for lemma in self.matrices:
            self.transform(lemma, fn)

    def resetSize(self, kCells, newValue=0):
        self.kCells = kCells

        for lem, sub in self.lem_2_wf.items():
            nWfs = len(sub)
            self.matrices[lem] = np.zeros( (nWfs, self.kCells) )
            if newValue != 0:
                self.matrices[lem][:] = newValue
