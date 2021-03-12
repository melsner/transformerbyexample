from __future__ import division, print_function
from sys import argv, exit, stderr, stdout
import os
from sklearn import manifold
from sklearn.cluster import KMeans
import numpy as np
from segment_functions import *
from matching import *
from clustering import Clustering, ResponsibilityMatrices
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
import random
from collections import defaultdict, Counter

import six
if six.PY2:
    import cPickle as pickle
else:
    import pickle


def record_ST_split(args, f, POS_list):

    lem_2_cell_2_wf, cell_2_wf, wf_2_cell = {}, {}, {}
    lem_2_wf, wf_2_lem = {}, {}

    # For each file in the split
    try:
        for line in open(f):
            if line.strip():
                lem, wf, cell_unfactored = line.strip().split("\t")
                if args.constrained_tag_set == None or cell_unfactored in args.constrained_tag_set:
                    POS = cell_unfactored.split(';')[0].split('.')[0]
                    try:
                        assert POS in ['N', 'ADJ', 'V']
                    except:
                        raise Exception('bad POS: {}'.format(POS))

                    # Record all instances of the desired POS
                    if POS in POS_list:

                        # dicts for unsupervised training, i.e., no cells
                        if wf not in wf_2_lem:
                            wf_2_lem[wf] = {}
                        wf_2_lem[wf][lem] = True

                        if lem not in lem_2_wf:
                            lem_2_wf[lem] = {}
                        lem_2_wf[lem][wf] = True 

                        # dicts for gold evaluation, i.e., include cells
                        if lem not in lem_2_cell_2_wf:
                            lem_2_cell_2_wf[lem] = {}
                        if cell_unfactored not in lem_2_cell_2_wf[lem]:
                            lem_2_cell_2_wf[lem][cell_unfactored] = {}
                        lem_2_cell_2_wf[lem][cell_unfactored][wf] = True

                        if cell_unfactored not in cell_2_wf:
                            cell_2_wf[cell_unfactored] = {}
                        cell_2_wf[cell_unfactored][wf] = True 

                        if wf not in wf_2_cell:
                            wf_2_cell[wf] = {}
                        wf_2_cell[wf][cell_unfactored] = True
    except FileNotFoundError:
        raise Exception('FileNotFoundError: No such file {}'.format(f))

    split = (lem_2_wf, wf_2_lem)
    gold = (lem_2_cell_2_wf, cell_2_wf, wf_2_cell)

    return split, gold


def filter_UM_by_raw_heldOut(args, UM_data_dir, ST_data_dir, raw_corpus, max_train_paradigms=None, max_train_instances=None, vocabSize=250):

    train, _, _, gold, POS_list = record_UM(args, UM_data_dir)
    (lem_2_wf, wf_2_lem) = train
    (lem_2_cell_2_wf, cell_2_wf, wf_2_cell) = gold
    origsize = sum(1 for lem in lem_2_wf for wf in lem_2_wf[lem])
    newsize = origsize

    if args.eval_extrinsic:
        train_sup, dev, test, gold_train_sup, gold_dev, gold_test, _ = record_ST(args, ST_data_dir)

        stdout.write('Removing held out data from training set..\n')

        for held_out in dev, test:
            (ho_lem_2_wf, ho_wf_2_lem) = held_out
            for wf in ho_wf_2_lem:
                if wf in wf_2_lem:

                    lems = list(wf_2_lem[wf])
                    cells = list(wf_2_cell[wf])

                    del wf_2_lem[wf]
                    del wf_2_cell[wf]

                    for lem in lems:
                        del lem_2_wf[lem][wf]
                        if len(lem_2_wf[lem]) == 0:
                            del lem_2_wf[lem]

                    for cell in cells:
                        del cell_2_wf[cell][wf]
                        if len(cell_2_wf[cell]) == 0:
                            del cell_2_wf[cell]

                    for lem in lems:
                        for cell in cells:
                            if cell in lem_2_cell_2_wf[lem] and wf in lem_2_cell_2_wf[lem][cell]:
                                del lem_2_cell_2_wf[lem][cell][wf]
                                if len(lem_2_cell_2_wf[lem][cell]) == 0:
                                    del lem_2_cell_2_wf[lem][cell]
                                if len(lem_2_cell_2_wf[lem]) == 0:
                                    del lem_2_cell_2_wf[lem]

        newsize = sum(1 for lem in lem_2_wf for wf in lem_2_wf[lem])

        (lem_2_wf_sup, wf_2_lem_sup) = train_sup

        #######################################################################################
        stdout.write('Shrunk data set from {} lem,wf tuples to {}\n'.format(origsize, newsize))
        #######################################################################################

    #######################################################################################
    stdout.write('Removing data unattested in raw corpus..\n')
    #######################################################################################

    stderr.write('Reading raw corpus *_* ')
    word_counts = Counter()
    corpus_forms = [] # corpus
    corpus_possible_lemmata = [] # lemmas
    vocab_all = {}
    OOV = 'OOV'
    lines = 0
    for line in open(raw_corpus):
        lines += 1
        if lines % 100000 == 0:
            stderr.write('{}K lines *_* '.format(str(int(lines/1000))))
            stderr.flush()
        for wf in line.split():
            corpus_forms.append(wf)
            word_counts[wf] += 1
            if wf in wf_2_lem:
                corpus_possible_lemmata.append(list(wf_2_lem[wf]))
                if wf not in vocab_all:
                    vocab_all[wf] = len(vocab_all)
            elif args.eval_extrinsic and wf in wf_2_lem_sup:
                corpus_possible_lemmata.append(list(wf_2_lem_sup[wf]))
                if wf not in vocab_all:
                    vocab_all[wf] = len(vocab_all)
            else:
                corpus_possible_lemmata.append(OOV)
    stderr.write('*_* Done reading raw corpus\n')

    wordCounts = Counter(corpus_forms)
    vocab = { "<->" : 0 }
    for wi, ci in wordCounts.most_common(vocabSize):
        #print("Adding word", wi, len(vocab))
        vocab[wi] = len(vocab)

    to_delete = []
    for wf in wf_2_lem:
        if wf not in vocab_all:
            to_delete.append(wf)
    for wf in to_delete:
        del wf_2_cell[wf]
        for lem in wf_2_lem[wf]:
            for cell in list(lem_2_cell_2_wf[lem]):
                if wf in lem_2_cell_2_wf[lem][cell]:
                    del lem_2_cell_2_wf[lem][cell][wf]
                    if len(lem_2_cell_2_wf[lem][cell]) == 0:
                        del lem_2_cell_2_wf[lem][cell]
                    if len(lem_2_cell_2_wf[lem]) == 0:
                        del lem_2_cell_2_wf[lem]
                    if wf in cell_2_wf:
                        del cell_2_wf[cell][wf]
                        assert len(cell_2_wf[cell]) > 0
            del lem_2_wf[lem][wf]
            if len(lem_2_wf[lem]) == 0:
                del lem_2_wf[lem]
        del wf_2_lem[wf]

    origsize = newsize
    newsize = sum(1 for lem in lem_2_wf for wf in lem_2_wf[lem])

    stdout.write('Shrunk data set from {} lem,wf tuples to {}\n'.format(origsize, newsize))

    threshold = None
    if max_train_instances != None and newsize > max_train_instances:
        threshold = max_train_instances
        metric = 'instances'
    elif max_train_paradigms != None and len(lem_2_wf) > max_train_paradigms:
        threshold = max_train_paradigms
        metric = 'paradigms'

    if threshold != None:
        ranked_paradigms = list((len(lem_2_wf[lem]), sum(list(word_counts[wf] for wf in lem_2_wf[lem])), lem) for lem in lem_2_wf)
        ranked_paradigms.sort(reverse=True)

        new_lem_2_wf, new_wf_2_lem, new_lem_2_cell_2_wf, new_wf_2_cell, new_cell_2_wf = {}, {}, {}, {}, {}
        total = 0
        while total < threshold:

            (attested_slots, lem_freq, lem) = ranked_paradigms.pop(0)
            if metric == 'paradigms':
                total += 1
            else:
                total += len(lem_2_wf[lem])
            new_lem_2_wf[lem] = dict(lem_2_wf[lem])
            new_lem_2_cell_2_wf[lem] = dict(lem_2_cell_2_wf[lem])                

            for cell in lem_2_cell_2_wf[lem]:
                if cell not in new_cell_2_wf:
                    new_cell_2_wf[cell] = {}
                for wf in lem_2_cell_2_wf[lem][cell]:
                    if wf not in new_wf_2_lem:
                        new_wf_2_lem[wf] = {}
                        new_wf_2_cell[wf] = {}
                    new_wf_2_lem[wf][lem] = True
                    new_wf_2_cell[wf][cell] = True
                    new_cell_2_wf[cell][wf] = True

        lem_2_wf, wf_2_lem, lem_2_cell_2_wf, wf_2_cell, cell_2_wf = new_lem_2_wf, new_wf_2_lem, new_lem_2_cell_2_wf, new_wf_2_cell, new_cell_2_wf

    if args.eval_extrinsic:
        #######################################################################################
        stdout.write('Adding shared task train data\n')
        #######################################################################################

        total = 0
        (lem_2_cell_2_wf_sup, cell_2_wf_sup, wf_2_cell_sup) = gold_train_sup

        for lem in lem_2_cell_2_wf_sup:
            if lem not in lem_2_cell_2_wf:
                lem_2_cell_2_wf[lem] = dict(lem_2_cell_2_wf_sup[lem])
            else:
                for cell in lem_2_cell_2_wf_sup[lem]:
                    if cell not in lem_2_cell_2_wf[lem]:
                        lem_2_cell_2_wf[lem][cell] = dict(lem_2_cell_2_wf_sup[lem][cell])
                    else:
                        for wf in lem_2_cell_2_wf_sup[lem][cell]:
                            lem_2_cell_2_wf[lem][cell][wf] = True

        for lem in lem_2_cell_2_wf:
            if lem not in lem_2_wf:
                lem_2_wf[lem] = {}
            for cell in lem_2_cell_2_wf[lem]:
                if cell not in cell_2_wf:
                    cell_2_wf[cell] = {}
                for wf in lem_2_cell_2_wf[lem][cell]:
                    if wf not in wf_2_lem:
                        wf_2_lem[wf] = {}
                    if wf not in wf_2_cell:
                        wf_2_cell[wf] = {}
                    lem_2_wf[lem][wf] = True
                    wf_2_lem[wf][lem] = True
                    cell_2_wf[cell][wf] = True
                    wf_2_cell[wf][cell] = True

    total = 0
    for lem in lem_2_wf:
        for wf in lem_2_wf[lem]:
            total += 1

    #######################################################################################
    stdout.write('Total wf,lem tuples: {}\n'.format(total))
    #######################################################################################

    if threshold != None:

        for i in range(len(corpus_possible_lemmata)):
            if corpus_possible_lemmata[i] != OOV:
                for lem in list(corpus_possible_lemmata[i]):
                    if lem not in lem_2_wf:
                        corpus_possible_lemmata[i].remove(lem)
                        if len(corpus_possible_lemmata[i]) == 0:
                            corpus_possible_lemmata[i] = OOV

    if max_train_paradigms != None:
        raise Exception('Have not implemented paradigm limitation on context filtered data yet..')

    train = (lem_2_wf, wf_2_lem)
    gold = (lem_2_cell_2_wf, cell_2_wf, wf_2_cell)
    
    if args.eval_extrinsic:
        return train, train_sup, dev, test, gold, gold_train_sup, gold_dev, gold_test, POS_list, corpus_forms, corpus_possible_lemmata, vocab, vocab_all
    else:
        return train, None, None, None, gold, None, None, None, POS_list, corpus_forms, corpus_possible_lemmata, vocab, vocab_all


def record_ST(args, data_dir, max_train_paradigms=None):

    if max_train_paradigms != None:
        raise NotImplementedError

    lem_2_cell_2_wf = {}
    cell_2_wf = {}
    wf_2_cell = {}
    gold = (lem_2_cell_2_wf, cell_2_wf, wf_2_cell)

    if args.POS == 'all':
        POS_list = ['N', 'V', 'ADJ']
    else:
        POS_list = [args.POS]

    if args.train_size == 'high':
        train_file = '{}-train-high'.format(os.path.join(data_dir, args.language))
    elif args.train_size == 'medium':
        train_file = '{}-train-medium'.format(os.path.join(data_dir, args.language))
    elif args.train_size == 'low':
        train_file = '{}-train-low'.format(os.path.join(data_dir, args.language))
    dev_file = '{}-dev'.format(os.path.join(data_dir, args.language))
    test_file = '{}-test'.format(os.path.join(data_dir, args.language))

    train, gold_train = record_ST_split(args, train_file, POS_list)
    dev, gold_dev = record_ST_split(args, dev_file, POS_list)
    test, gold_test = record_ST_split(args, test_file, POS_list)

    return train, dev, test, gold_train, gold_dev, gold_test, POS_list


def record_UM(args, data_dir, max_train_paradigms=None, max_train_instances=None):

    lem_2_cell_2_wf = {}
    lem_2_wf = {}
    wf_2_lem = {}
    cell_2_wf = {}
    wf_2_cell = {}

    if args.POS == 'all':
        POS_list = ['N', 'V', 'ADJ']
    else:
        POS_list = [args.POS]

    random.seed(0)
    lines = []
    for line in open(os.path.join(data_dir, args.language, args.language)):
        lines.append(line)
    random.shuffle(lines)
    total = 0
    for line in lines:
        if line.strip():
            lem, wf, cell_unfactored = line.strip().split("\t")

            go = False
            if max_train_paradigms == max_train_instances == None:
                go = True
            elif lem in lem_2_wf:
                go = True
            elif max_train_paradigms != None and len(lem_2_wf) < max_train_paradigms:
                go = True
            elif max_train_instances != None and total < max_train_instances:
                go = True

            if go:
                POS = cell_unfactored.split(';')[0].split('.')[0]

                if args.constrained_tag_set == None or cell_unfactored in args.constrained_tag_set:
                    try:
                        assert POS in ['N', 'ADJ', 'V']
                    except:
                        raise Exception('bad POS: {}'.format(POS))

                    # Record all instances of the desired POS
                    if POS in POS_list:

                        # dicts for unsupervised training, i.e., no cells
                        if wf not in wf_2_lem:
                            wf_2_lem[wf] = {}
                        wf_2_lem[wf][lem] = True

                        if lem not in lem_2_wf:
                            lem_2_wf[lem] = {}
                        if wf not in lem_2_wf[lem]:
                            total += 1
                            lem_2_wf[lem][wf] = True 

                        # dicts for gold evaluation, i.e., include cells
                        if lem not in lem_2_cell_2_wf:
                            lem_2_cell_2_wf[lem] = {}
                        if cell_unfactored not in lem_2_cell_2_wf[lem]:
                            lem_2_cell_2_wf[lem][cell_unfactored] = {}
                        lem_2_cell_2_wf[lem][cell_unfactored][wf] = True

                        if cell_unfactored not in cell_2_wf:
                            cell_2_wf[cell_unfactored] = {}
                        cell_2_wf[cell_unfactored][wf] = True 

                        if wf not in wf_2_cell:
                            wf_2_cell[wf] = {}
                        wf_2_cell[wf][cell_unfactored] = True

    if max_train_instances != None:
        lems = list(lem_2_wf)
        random.seed(10)
        random.shuffle(lems)

        while total > max_train_instances:
            lem = lems.pop(0)
            for wf in lem_2_wf[lem]:
                total -= 1
                del wf_2_lem[wf][lem]
                if len(wf_2_lem[wf]) == 0:
                    del wf_2_lem[wf]
            del lem_2_wf[lem]
            del lem_2_cell_2_wf[lem]

        cell_2_wf = {}
        wf_2_cell = {}
        for lem in lem_2_cell_2_wf:
            for cell in lem_2_cell_2_wf[lem]:
                if cell not in cell_2_wf:
                    cell_2_wf[cell] = {}
                for wf in lem_2_cell_2_wf[lem][cell]:
                    if wf not in wf_2_cell:
                        wf_2_cell[wf] = {}
                    wf_2_cell[wf][cell] = True
                    cell_2_wf[cell][wf] = True

    train = (lem_2_wf, wf_2_lem)
    gold = (lem_2_cell_2_wf, cell_2_wf, wf_2_cell)
    dev = test = {}, {}

    return train, dev, test, gold, POS_list


def get_exponents(lem_2_wf, nCells):

    lem_2_base = {}

    exponent_2_lem = {}
    lem_2_wf_2_exponent = {}
    base_2_form_2_exponent = {'problem_index':[]}

    base_id = -1
    for lem in lem_2_wf:
        base_id += 1
        forms = tuple(lem_2_wf[lem].keys())
        base_2_form_2_exponent[base_id] = {}
        uForms = list(set(forms))
        lem_2_wf_2_exponent[lem] = {}

        ### Store all single-form paradigms to address later
        if len(uForms) <= 1:
            base_2_form_2_exponent['problem_index'].append(base_id)
            base_2_form_2_exponent[base_id] = (lem, forms)

        ### We can get bases and exponents ulti-form paradigms now
        else:
            base = lcs(uForms)
            exponent = getExponent(base, uForms)
            base_key = '{}_{}'.format(str(base_id), base)
            lem_2_base[lem] = base_key
            base_2_form_2_exponent[base_id][base_key] = {}

            for ind in range(len(exponent)):
                exp = exponent[ind]
                wf = uForms[ind]
                lem_2_wf_2_exponent[lem][wf] = exp
                base_2_form_2_exponent[base_id][base_key][wf] = exp

                if exp not in exponent_2_lem:
                    exponent_2_lem[exp] = {}
                exponent_2_lem[exp][lem] = True

    return exponent_2_lem, lem_2_wf_2_exponent, base_2_form_2_exponent, lem_2_base


def embed_distMat(dist_matrix, dim):

    stdout.write('\tEmbedding and clustering the matrix\n')
    mds = manifold.MDS(n_components=dim, dissimilarity="precomputed")
    results = mds.fit(dist_matrix)
    posMat = results.embedding_

    return posMat


def get_signature_cooccurrences(exponent_2_lem):

    stdout.write('\tLearning a distance matrix from signature co-occurrences\n')
    all_exponents = {}
    exponents_2_cooccurences = {}
    eligible = dict(exponent_2_lem)

    for exponent1 in exponent_2_lem:
        all_exponents[exponent1] = len(exponent_2_lem[exponent1])
        del eligible[exponent1]
        if exponent1 not in exponents_2_cooccurences:
            exponents_2_cooccurences[exponent1] = {}
        for exponent2 in eligible:
            if exponent2 not in exponents_2_cooccurences:
                exponents_2_cooccurences[exponent2] = {}
            exponents_2_cooccurences[exponent1][exponent2] = exponents_2_cooccurences[exponent2][exponent1] = len(set(exponent_2_lem[exponent1]).intersection(set(exponent_2_lem[exponent2])))

    ranked_exponents = list((x, y) for (y, x) in all_exponents.items())
    ranked_exponents.sort(reverse=True)
    exponent_freq_list = [x[1] for x in ranked_exponents]

    return exponents_2_cooccurences, exponent_freq_list


def cluster_map(exponent_embeddings, k, exponent_list):

    if len(exponent_embeddings) < k:
        stderr.write('WE IDENTIFIED FEWER EXPONENTS ({}) THAN SPECIFIED CELLS ({})\n\tIs there like a ton of syncretism going on or something???\n\tNo matter, we will just collapse the syncretisms we predict..\n'.format(str(len(exponent_embeddings)), str(k)))
        k = len(exponent_embeddings)
    kmeans = KMeans(n_clusters=k).fit(exponent_embeddings)
    ind_cluster = kmeans.labels_
    cluster_2_centroid = kmeans.cluster_centers_

    ### RECORD AFFIX TO CELL MAPPINGS
    exponent_2_cluster = {}
    for ind in range(len(ind_cluster)):
        cluster = ind_cluster[ind]
        exponent = exponent_list[ind]
        exponent_2_cluster[exponent] = cluster

    return k, exponent_2_cluster, cluster_2_centroid


def rank_exponents_by_likelihood(exponent_list, lem_2_wf_2_exponent):

    stdout.write('\tRanking exponents by their likelihoods\n')
    ngram_2_exponent = {}
    for exponent in exponent_list:
        ngram_2_exponent[exponent] = {}

    for lem in lem_2_wf_2_exponent:
        for wf in lem_2_wf_2_exponent[lem]:
            actual_exponent = lem_2_wf_2_exponent[lem][wf]
            for possible_exponent in exponent_list:
                could_work = True
                form = '<{}>'.format(wf)
                for a in possible_exponent:
                    if a in form:
                        form = form.split(a, 1)[1]
                    else:
                        could_work = False
                        break
                if could_work:
                    if actual_exponent not in ngram_2_exponent[possible_exponent]:
                        ngram_2_exponent[possible_exponent][actual_exponent] = 0
                    ngram_2_exponent[possible_exponent][actual_exponent] += 1

    exponent_2_prob = {}
    for exponent in ngram_2_exponent:
        denominator = 0
        for actual_exponent in ngram_2_exponent[exponent]:
            denominator += ngram_2_exponent[exponent][actual_exponent]
        exponent_2_prob[exponent] = ngram_2_exponent[exponent][exponent] / denominator

    ranked_affixes = list((x, y) for (y, x) in exponent_2_prob.items())
    ranked_affixes.sort(reverse=True)
    exponent_prob_list = [x[1] for x in ranked_affixes]

    return exponent_prob_list


def get_exponents_for_singleton_paradigms(lem_2_wf_2_exponent, ranked_exponents_freq, ranked_exponents_prob, base_2_form_2_exponent, lem_2_base):
    
    stdout.write('\tGetting most likely exponents for single-form paradigms\n')
    for base_id in base_2_form_2_exponent['problem_index']:
        (lem, forms) = base_2_form_2_exponent[base_id]
        uForms = list(set(forms))
        base_2_form_2_exponent[base_id] = {}

        assert len(uForms) == 1
        assert lem_2_wf_2_exponent[lem] == {}

        best_exponent = ranked_exponents_freq[0]
        best_stem = uForms[0]
        for exponent in ranked_exponents_prob:
            form = '<{}>'.format(uForms[0])
            base_key = '{}_'.format(str(base_id))
            match = True
            for a in exponent:
                if a in form:
                    base_key += form.split(a, 1)[0]
                    form = form.split(a, 1)[1]
                else:
                    match = False
                    break
            if match:
                base_key += form.replace('>', '')
                best_exponent = exponent
                best_stem = base_key.replace('<', '')
                break

        base_2_form_2_exponent[base_id][best_stem] = {uForms[0]:best_exponent}
        lem_2_wf_2_exponent[lem][uForms[0]] = best_exponent
        lem_2_base[lem] = best_stem

    del base_2_form_2_exponent['problem_index']

    return lem_2_wf_2_exponent, base_2_form_2_exponent, lem_2_base


def vec_cos_sim(v1, v2):

    cos_sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]

    return cos_sim


def build_distMat(ranked_exponents_freq, exponents_2_cooccurences):

    dist_matrix = np.array([[0.0]*len(ranked_exponents_freq)]*len(ranked_exponents_freq))
    for rowInd in range(len(dist_matrix)):
        exponent1 = ranked_exponents_freq[rowInd]
        for columnInd in range(rowInd+1, len(dist_matrix)):
            exponent2 = ranked_exponents_freq[columnInd]
            cooccurence_dist = 0.0
            if exponent1 in exponents_2_cooccurences and exponent2 in exponents_2_cooccurences[exponent1]:
                cooccurence_dist = exponents_2_cooccurences[exponent1][exponent2]
            dist_matrix[rowInd][columnInd] = dist_matrix[columnInd][rowInd] =cooccurence_dist

    return dist_matrix


def map_forms_to_clusters(k, base_2_wf_2_exponent, exponent_2_embeddings, cluster_2_centroid, syncretism_penalty, overabundance_penalty, assign_method, rand_init=False):

    stdout.write('\tMapping all (base, form) tuples to one or more clusters\n')
    # build responsibility matrices for all lemmata
        # Compile into responsibility matrix over all (base, wf) tuples
    # Each row approximates P(form|cell) as cosine_sim(form's_exponent's_embedding, cell's_cluster's_centroid)
    # Hungarian map each lemma individually with kilimanjaro assignment method
        # discourages over abundance and syncretism while gaurunteeing all input nodes are mapped
        # extent of discouragement reflected by two independently tunable parameters

    map_score = 0
    clustering = Clustering(base_2_wf_2_exponent, k)
    responsibilities = ResponsibilityMatrices(base_2_wf_2_exponent, k)

    signatures_2_member_count = {}
    base_2_signatures = {}
    # for cluster in range(k): RM
    #     cluster_2_base_2_wf[cluster] = {}
    #     cluster_2_wf[cluster] = {}

    cache = {} # cache
    # Go through all bases
    for base in base_2_wf_2_exponent:
        wfs = list(base_2_wf_2_exponent[base].keys())
        base_2_signatures[base] = {}

        # Initialize paradigm responsibilities
        
        # Go through all words, updating responsibility matrix row IDs
        for wfi in range(len(wfs)):
            wf = wfs[wfi]

            # Get the exponent and relevant embedding
            exp = base_2_wf_2_exponent[base][wf]
            wf_embedding = exponent_2_embeddings[exp]

            # Get cosine similarity of exponent and cluster centroid for each cluster
            for clusterj in range(k):
                # check if we've already cached the cosine similarity
                if exp in cache and clusterj in cache[exp]:
                    similarity = cache[exp][clusterj]
                else:
                    cluster_centroid = cluster_2_centroid[clusterj]
                    if exp not in cache:
                        cache[exp] = {}
                    # calculate cosine similarity cache
                    cache[exp][clusterj] = (vec_cos_sim(wf_embedding, cluster_centroid))# + 1.0) / 2.0
                # update responsibilities matrix
                responsibilities.assign(base, wf, clusterj, cache[exp][clusterj])

        # translate from cos sim to probs via softmax
        responsibilities.transform(base, softmax)

        # Do bipartite mapping on the paradigm's responsibility matrix
        G = bipartite_match(responsibilities.matrices[base], row_penalty=syncretism_penalty, column_penalty=overabundance_penalty, assign_method=assign_method)
        map_score += G.get_score()

        ### hard assignment forms to clusters/mixture components
        signatures = [[None]*k]
        # random assignment
        if rand_init:
            eligible_clusters = list(range(k))
            eligible_clusters *= int(len(wfs)/k)+1
            used_clusters = {}
            for wfi in range(len(wfs)):
                wf = wfs[wfi]
                cluster = eligible_clusters.pop(random.randint(0, len(eligible_clusters)-1))
                if cluster in used_clusters:
                    # sigs
                    for sid in range(len(signatures)):
                        signatures.append(signatures[sid])
                        signatures[-1][cluster] = exponent
                    # * #
                else:
                    used_clusters[cluster] = True
                    # sigs
                    for sid in range(len(signatures)):
                        signatures[sid][cluster] = exponent
                    # * #
                clustering.addToCluster(base, wf, cluster) #overabundance is allowed so don't "move"

        # optimal mapping determined by G
        else:
            used_clusters = {}
            for wfi in range(len(wfs)):
                wf = wfs[wfi]
                exponent = base_2_wf_2_exponent[base][wf]
                for cluster in G.assignments[wfi].keys():
                    if cluster in used_clusters:
                        # sigs
                        for sid in range(len(signatures)):
                            signatures.append(signatures[sid])
                            signatures[-1][cluster] = exponent
                        # * #
                    else:
                        used_clusters[cluster] = True
                        # sigs
                        for sid in range(len(signatures)):
                            signatures[sid][cluster] = exponent
                        # * #
                    clustering.addToCluster(base, wf, cluster) #overabundance is allowed so don't "move"
        # sigs
        for signature in signatures:
            signature = tuple(signature)
            base_2_signatures[base][signature] = True
            if signature not in signatures_2_member_count:
                signatures_2_member_count[signature] = 0
            signatures_2_member_count[signature] += 1
        # * #

    return map_score, responsibilities, clustering, signatures_2_member_count, base_2_signatures


def take_first_second(elem):
    return (elem[0], elem[1])


def initialize(train, k, supervision, embed_dim=200, syncretism_penalty=1.0, overabundance_penalty=1.0, assign_method='kilimanjaro', rand_init=False, latent_IC=True, seed=0, max_sigs=50):

    random.seed(seed)

    if 'cell' in supervision:
        raise Exception('\nSorry, we have not yet implemented the semi supervised version where some of the cell labels are known.')

    elif len(supervision) == 0:
        raise Exception('\nSorry, we have not yet implemented the fully unsupervised version that treats both paradigm membership and lemmata/bases as unobserved latent variables.')

    else:

        lem_2_wf, wf_2_lem = train

        ### TRACK INTRA-SIGNATURE AFFIX CO-OCCURENCES
        exponent_2_lem, lem_2_wf_2_exponent, base_2_form_2_exponent, lem_2_base = get_exponents(lem_2_wf, k)
        assert len(base_2_form_2_exponent) == len(lem_2_wf) + 1
        stdout.write('\tPredicted {} exponents\n'.format(len(exponent_2_lem)))
                 
        ### TALLY THE INTRA-SIGNATURE CO-OCCURRENCES OF ALL EXPONENTS
        exponents_2_cooccurences, ranked_exponents_freq = get_signature_cooccurrences(exponent_2_lem)
        assert len(exponent_2_lem) == len(exponents_2_cooccurences)

        ### BUILD A DISTANCE MATRIX
        dist_matrix = build_distMat(ranked_exponents_freq, exponents_2_cooccurences)

        ### EMBED THE DISTANCE MATRIX
        exponent_embeddings = embed_distMat(dist_matrix, min(embed_dim, len(dist_matrix)))

        ### CLUSTER THE EMBEDDED AFFIXES INTO CELLS
        k, exponent_2_cluster, cluster_2_centroid = cluster_map(exponent_embeddings, k, ranked_exponents_freq)

        ### RANK AFFIX_LIST BY HOW LIKELY AN AFFIX IS GIVEN THAT IT OCCURS AS AN NGRAM
        ranked_exponents_prob = rank_exponents_by_likelihood(ranked_exponents_freq, lem_2_wf_2_exponent)

        ### ADD IN SINGLE-FORM LEMMAS
        lem_2_wf_2_exponent, base_2_form_2_exponent, lem_2_base = get_exponents_for_singleton_paradigms(lem_2_wf_2_exponent, ranked_exponents_freq, ranked_exponents_prob, base_2_form_2_exponent, lem_2_base)
        assert len(base_2_form_2_exponent) == len(lem_2_wf)

        ### DECIDE IF WE ARE ALLOWED TO SEE EXPLICIT LEMMAS
        if 'lemma' in supervision:
            for lem in lem_2_base:
                lem_2_base[lem] = lem
            base_2_wf_2_exponent = lem_2_wf_2_exponent
        else:
            base_2_wf_2_exponent = {}
            for ID in base_2_form_2_exponent:
                for base in base_2_form_2_exponent[ID]:
                    base_2_wf_2_exponent[base] = dict(base_2_form_2_exponent[ID][base])

        ### MAP EXPONENTS TO EMBEDDINGS
        exponent_2_embeddings = {}
        for i in range(len(ranked_exponents_freq)):
            exponent_2_embeddings[ranked_exponents_freq[i]] = exponent_embeddings[i]

        ### MAP FORMS TO CLUSTERS
        #syncretisms, overabundances, map_score, base_2_responsibilities, base_2_row_2_wf, wf_2_cluster, cluster_2_wf, cluster_2_base_2_wf, base_2_wf_2_cluster, signatures_2_member_count, base_2_signatures = map_forms_to_clusters(k, base_2_wf_2_exponent, exponent_2_embeddings, cluster_2_centroid, syncretism_penalty, overabundance_penalty, assign_method, rand_init=rand_init) #RM
        map_score, responsibilities, clustering, signatures_2_member_count, base_2_signatures = map_forms_to_clusters(k, base_2_wf_2_exponent, exponent_2_embeddings, cluster_2_centroid, syncretism_penalty, overabundance_penalty, assign_method, rand_init=rand_init)

        ### INITIALIZE LATENT INFLECTION CLASS VARIABLES WITH SIGNATURES
        base_2_IC = None
        if latent_IC:
            ranked_sigs = []
            for sig in signatures_2_member_count:
                ranked_sigs.append((sum(1 for x in sig if x != None), signatures_2_member_count[sig], sig))
            ranked_sigs.sort(key=take_first_second, reverse=True)
            ranked_sigs = list(l[2] for l in ranked_sigs)

            stdout.write('\tReducing {} predicted micro-classes to {} inflection classes\n'.format(len(ranked_sigs), max_sigs))
            ### Merge microclasses until we get a manageable number of signatures
            allSigs_2_final_limited_sigs = {}
            for rsi in range(len(ranked_sigs)):
                if rsi < max_sigs:
                    allSigs_2_final_limited_sigs[rsi] = rsi
                else:
                    signature = ranked_sigs[rsi]
                    max_score = -1
                    cieling_score = sum(1 for x in signature if x != None)
                    for fsi in range(max_sigs):
                        fin_signature = ranked_sigs[fsi]

                        score = sum(1 for x in range(len(signature)) if signature[x] != None and signature[x] == fin_signature[x])
                        if score > max_score:
                            new_sig_i = fsi
                            max_score = score
                            if score == cieling_score:
                                break
                    allSigs_2_final_limited_sigs[rsi] = new_sig_i

            ### assign bases to ICs
            base_2_IC = {}
            for base in base_2_signatures:
                orig_sig_i = min(list(ranked_sigs.index(sig) for sig in base_2_signatures[base]))
                base_2_IC[base] = allSigs_2_final_limited_sigs[orig_sig_i]

        ### REPORT INITIALIZATION STATISTICS
        if latent_IC:
            stdout.write('\t\tConstraining ICs to not exceed {}\n'.format(len(set(allSigs_2_final_limited_sigs.values()))))
        stdout.write('\tPredicted {} syncretisms and {} overabundances\n'.format(clustering.syncretisms(), clustering.overabundances()))
        # stdout.write('\tInitial mapping scored {}\n'.format(str(round(map_score, 2))))
        # stdout.write('\t\t(Initialization map score not comparable to EM map scores)\n'.format(str(round(map_score, 2))))

        return responsibilities, clustering, base_2_IC, base_2_wf_2_exponent, lem_2_base


def initialize_context_model(context_forms, context_possible_lemmata, context_vocab, base_2_wf_2_exponent, lem_2_base=None, genMorph=True, exponent_vocab=True, OOV='OOV', window=1, nExes=1):

    lim = 2*len(context_vocab)
    for ind in range(len(context_possible_lemmata)):
        possible_lemmata = context_possible_lemmata[ind]
        wf = context_forms[ind]
        if possible_lemmata != OOV:
            base = random.choice(possible_lemmata)
            if lem_2_base != None:
                if base not in lem_2_base:
                    base = OOV
                else:
                    base = lem_2_base[base]
            context_possible_lemmata[ind] = base
            if base not in base_2_wf_2_exponent or wf not in base_2_wf_2_exponent[base]:
                context_possible_lemmata[ind] = OOV
            else:
                if base_2_wf_2_exponent[base][wf] not in context_vocab and len(context_vocab) < lim:
                    context_vocab[base_2_wf_2_exponent[base][wf]] = len(context_vocab)
    context_possible_bases = context_possible_lemmata

    assert len(context_possible_bases) == len(context_forms)
        
    def padLst(lst):
        while len(lst) < nExes:
            lst.append("<->")

    def selectExes(lst):
        random.shuffle(lst)
        del lst[nExes:]
        
    instances = []
    for ind in range(len(context_forms)):
        base = context_possible_bases[ind]
        wf = context_forms[ind]

        if base != OOV: # and base in base_2_wf_2_exponent and wf in base_2_wf_2_exponent[base]:

            posL = []
            posR = []
            morphL = []
            morphR = []
            verbose = False

            # Generating tokens around the lemma

            for iLeft in range(ind - window, ind):
                if iLeft >= 0:
                    lForm = context_forms[iLeft]
                    lBase = context_possible_bases[iLeft]
                    if lForm in context_vocab:
                        posL.append(lForm)
                    if genMorph and lBase != OOV and exponent_vocab: 
                            morphL.append(base_2_wf_2_exponent[lBase][lForm])

            for iRight in range(ind + 1, ind + window + 1):
                if iRight < len(context_forms):

                    rForm = context_forms[iRight]
                    rBase = context_possible_bases[iRight]
                    if rForm in context_vocab:
                        posR.append(rForm)
                    if genMorph and rBase != OOV and exponent_vocab: 
                        morphR.append(base_2_wf_2_exponent[rBase][rForm])

            for lst in [posL, posR, morphL, morphR]:
                padLst(lst)
                selectExes(lst)

            instances.append([base, wf, posL, posR, morphL, morphR])

    all_contexts = [ [], [], [], [] ]
    for inst in instances:
        for ii, ci in enumerate(inst[2:]):
            for pi in ci:
                if pi != "<->":
                    all_contexts[ii].append(pi)

    for ci in all_contexts:
        random.shuffle(ci) #generate negative exes using permutation

    # return all_contexts, instances, context_vocab

    def genNeg(pos, contexts):
        npos = sum([1 for xx in pos if xx != "<->"])
        neg = []
        for ii in range(npos):
            neg.append(contexts.pop())
        while len(neg) < nExes:
            neg.append("<->")
        return neg
        
    for ind in range(len(instances)):
        inst = instances[ind]
        negs = []
        for ii, ci in enumerate(inst[2:]):
            negs.append(genNeg(ci, all_contexts[ii]))
        instances[ind] = instances[ind] + negs

    return instances, context_vocab







