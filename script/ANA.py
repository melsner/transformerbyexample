from __future__ import division, print_function
from sys import argv, exit, stderr, stdout
import os
from sklearn import manifold
from sklearn.cluster import KMeans
from scipy.special import softmax
import numpy as np
import argparse
from segment_functions import *
from initialize import *
from matching import *
from inflector import Inflector
import multiprocessing as mp
import random
import math
import six
if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl
from Seq2seq import seq2seq_runner, dataloader
from Seq2seq import model as model_lib
from context_vectors import get_context_vectors

####################################################################################
### Classes
####################################################################################

class Data_manager:

    def __init__(self, args, seed=0):

        random.seed(seed)

        if args.benchmark_only:
            args.eval_extrinsic = True
        self.max_train_paradigms = args.max_train_paradigms
        self.max_train_instances = args.max_train_instances
        self.supervision = args.supervision
        self.eval_extrinsic = args.eval_extrinsic
        if args.model_context:
            args.filter_data = True

        if args.filter_data:
            self.train, self.train_sup, self.dev, self.test, self.gold, self.gold_train_sup, self.gold_dev, self.gold_test, self.POS_list, self.context_forms, self.context_possible_lemmata, self.context_vocab, self.context_vocab_all = filter_UM_by_raw_heldOut(args, args.UMdir, args.STdir, args.raw_corpus, max_train_paradigms=self.max_train_paradigms, max_train_instances=self.max_train_instances)
        elif args.UMdir != None:
            assert args.STdir == None and args.UDdir == None 
            self.train, self.dev, self.test, self.gold, self.POS_list = record_UM(args, args.UMdir, max_train_paradigms=self.max_train_paradigms, max_train_instances=self.max_train_instances)
        elif args.STdir != None:
            assert args.UMdir == None and args.UDdir == None
            self.train, self.dev, self.test, self.gold, self.gold_dev, self.gold_test, self.POS_list = record_ST(args, args.STdir, max_train_paradigms=self.max_train_paradigms)

        self.lem_2_cell_2_wf, self.cell_2_wf, self.wf_2_cell = self.gold
        self.base_2_wf, self.wf_2_base = self.train

        self.ORACLE_K = len(self.cell_2_wf)
        self.BLIND_K = 0
        for base in self.base_2_wf:
            self.BLIND_K = max(self.BLIND_K, len(self.base_2_wf[base]))

        try:
            self.k = int(args.k)
        except ValueError:
            if args.k.lower() == 'blind':
                self.k = self.BLIND_K
            elif args.k.lower() == 'oracle':
                self.k = self.ORACLE_K
            else:
                raise Exception('{} is not a supported option for number of cell clusters. Please set k argument to "blind", "oracle", or an integer'.format(args.k))

        self.language = args.language
        self.constrained_tag_set = args.constrained_tag_set
        if args.constrained_tag_set != None:
            self.cell_constraints = '_'.join(self.constrained_tag_set)
        else:
            self.cell_constraints = '_'.join(self.POS_list)

        self.MODEL_DIR = os.path.join(args.model_location, '{}_{}'.format('-'.join(args.language.split()), self.cell_constraints.replace(';', '-')))
        if not os.path.isdir(self.MODEL_DIR):
            os.makedirs(self.MODEL_DIR)
        else:
            raise Exception('Model directory {} already exists'.format(self.MODEL_DIR))

    def print_data_summary(self):

        stdout.write("\n\n######################################################\n### DATA SUMMARY\n######################################################\nLANGUAGE: {}\t\tPOS/TAGS: {}\n".format(self.language, self.cell_constraints.replace('_', ' ')))
        stdout.write('SUPERVISION: {}\n'.format(', '.join(self.supervision)))
        stdout.write('Oracle K: {}\nBlind K: {}\nModel K: {}\n'.format(str(self.ORACLE_K), str(self.BLIND_K), str(self.k)))
        stdout.write('\nGold Lemmata: {}\nGold Forms: {}\nTrain Paradigms: {}\nTrain Forms: {}\n######################################################\n'.format(str(len(self.lem_2_cell_2_wf)), str(len(self.wf_2_cell)), str(len(self.base_2_wf)), str(len(self.wf_2_base))))
        stdout.flush()


class Analyzer():

    def __init__(self, args, D, seed=0):

        stdout.write('\nINITIALIZING HARD EM\n'.format(D.k))
        stdout.flush()
        random.seed(seed)

        # self.reconsider_cluster_count = True
        # try:
        #     if args.k.lower() == 'oracle':
        #         self.reconsider_cluster_count = False 
        # except:
        #     pass
        self.eval_extrinsic = args.eval_extrinsic
        self.benchmark_only = args.benchmark_only
        self.greed = args.ana_greed # 0.5
        self.D = D
        self.rand_init = args.ana_rand_init
        self.estop = args.ana_estop
        self.assign_method = args.assign_method
        self.syncretism_penalty = args.syncretism_penalty
        self.overabundance_penalty = args.overabundance_penalty
        self.supervision = args.supervision
        self.MODEL_DIR = D.MODEL_DIR
        self.latent_IC = args.ana_latent_IC
        self.model_context = args.model_context
        if self.latent_IC:
            self.tune_latent_IC = args.ana_tune_latent_IC
        else:
            self.tune_latent_IC = False
        self.IC_patience = args.ana_IC_patience
        self.tunable_penalties = args.ana_tunable_penalties
        ## surface form signature prediction initialization

        self.base_2_responsibilities, self.base_2_row_2_wf, self.wf_2_cluster, self.cluster_2_wf, self.cluster_2_base_2_wf, self.base_2_wf_2_cluster, self.base_2_IC, self.base_2_wf_2_exponent, self.lem_2_base = initialize(self.D.train, self.D.k, self.supervision, embed_dim=args.embed_dim, syncretism_penalty=self.syncretism_penalty, overabundance_penalty=self.overabundance_penalty, assign_method=self.assign_method, rand_init=self.rand_init, latent_IC=self.latent_IC, seed=seed)

        self.base_2_IC_consistency = {}
        self.undecided_baseIC = {}
        if self.base_2_IC != None:
            for base in self.base_2_IC:
                self.base_2_IC_consistency[base] = 0
                self.undecided_baseIC[base] = True

        ##
        self.k = len(self.cluster_2_wf)
        self.base_2_penalties = {}
        for base in self.base_2_wf_2_cluster:
            self.base_2_penalties[base] = [self.syncretism_penalty, self.overabundance_penalty]
            # syncretism_penalty_vector = [self.syncretism_penalty]*self.k
            # self.base_2_penalties[base] = [syncretism_penalty_vector, self.overabundance_penalty]

        self.purity = purityEval(self.cluster_2_wf, self.wf_2_cluster, self.D.cell_2_wf, self.D.wf_2_cell,
        msg='EVALUATING PURITY OF K-MEANS INITIALIZATION')
        ##
        self.test_set = {}
        self.still_eligible = {}
        for base in self.base_2_wf_2_cluster:
            for wf in self.base_2_wf_2_cluster[base]:
                self.still_eligible[(base, wf)] = {}
                for cluster in range(self.k):
                    self.still_eligible[(base, wf)][cluster] = True
                    if args.ana_latent_IC:
                        self.test_set[(base, wf, (cluster, self.base_2_IC[base], None))] = True
                    else:
                        self.test_set[(base, wf, (cluster, None, None))] = True

        if self.model_context:
            lem_2_base = dict(self.lem_2_base)
            if 'lemma' in self.supervision:
                lem_2_base = None
            self.context_instances, self.context_vocab = initialize_context_model(D.context_forms, D.context_possible_lemmata, D.context_vocab, self.base_2_wf_2_exponent, lem_2_base=lem_2_base, exponent_vocab=args.exponent_vocab)

        self.round = -1
        self.stop = False
        self.retrain_from_scratch = args.ana_retrain_from_scratch
        self.max_s2s_dev_size = args.s2s_max_dev_size
        self.map_scores = []
        self.best_map_score = None
        self.max_rounds = args.ana_max_rounds
        self.debug = args.ana_debug
        self.separateBases = True
        self.parallel_partitions = args.ana_parallel_partitions

        # ### initialize s2s models
        # np.random.seed(seed)
        # self.fold_2_s2s = {'a': Inflector(args, os.path.join(self.MODEL_DIR, 'Inflection_a')), 'b': Inflector(args, os.path.join(self.MODEL_DIR, 'Inflection_b'))}
        # self.fold_2_lemmatizer = {'a': Inflector(args, os.path.join(self.MODEL_DIR, 'Lemmatization_a')), 'b': Inflector(args, os.path.join(self.MODEL_DIR, 'Lemmatization_b'))}
        # if self.benchmark_only:
        #     self.benchmark_random = Inflector(args, os.path.join(self.MODEL_DIR, 'Benchmark_random'), extrinsic=True)
        #     self.benchmark_initialization = Inflector(args, os.path.join(self.MODEL_DIR, 'Benchmark_initialization'), extrinsic=True)
        #     self.benchmark_oracle = Inflector(args, os.path.join(self.MODEL_DIR, 'Benchmark_oracle'), extrinsic=True)
        # else:
        #     self.extrinsic_inflection = Inflector(args, os.path.join(self.MODEL_DIR, 'Extrinsic_inflection'), extrinsic=True)

        ### initialize s2s models
        np.random.seed(seed)
        os.makedirs(os.path.join(self.MODEL_DIR, 'Inflection'))
        os.makedirs(os.path.join(self.MODEL_DIR, 'Lemmatization'))
        self.fold_2_s2s = {'a': None, 'b': None}
        self.fold_2_lemmatizer = {'a': None, 'b': None}
        if self.benchmark_only:
            self.benchmark_random = Inflector(args, os.path.join(self.MODEL_DIR, 'Benchmark_random'), extrinsic=True)
            self.benchmark_initialization = Inflector(args, os.path.join(self.MODEL_DIR, 'Benchmark_initialization'), extrinsic=True)
            self.benchmark_oracle = Inflector(args, os.path.join(self.MODEL_DIR, 'Benchmark_oracle'), extrinsic=True)
        else:
            self.extrinsic_inflection = Inflector(args, os.path.join(self.MODEL_DIR, 'Extrinsic_inflection'), extrinsic=True)
     
    def run_EM(self):

        ### Save the initialized model
        pkl.dump(self.base_2_responsibilities, open(os.path.join(self.MODEL_DIR, 'step_initialized_responsibilities.pkl'), 'wb' )) 
        if self.benchmark_only:
            self.stop = True

        ### Run EM until convergence
        while not self.stop:
            self.round += 1
            stderr.write('\nSTEP: {}\n'.format(self.round))

            if self.model_context:
                self.wf_2_context_embedding, self.exponent_2_context_embedding, self.cluster_2_context_embedding, full_context_matrix, wf_cluster_or_exp_2_embedding_row = get_context_vectors(self.base_2_wf_2_cluster, self.cluster_2_base_2_wf, self.context_instances, self.context_vocab)

                # self.wf_context_matrix = []
                self.cluster_context_matrix = []
                for cluster in range(self.k):
                    self.cluster_context_matrix.append(self.cluster_2_context_embedding[cluster])
                self.cluster_context_matrix = np.array(self.cluster_context_matrix)

                # update data
                for tup in list(self.test_set):
                    (base, wf, (cluster, IC, _)) = tup
                    newTup = (base, wf, (cluster, IC, cluster))
                    del self.test_set[tup]
                    self.test_set[newTup] = True

                # load context embeddings
                self.fold_2_s2s['a'].load_preTrained_context_embeddings(self.cluster_context_matrix)
                self.fold_2_s2s['b'].load_preTrained_context_embeddings(self.cluster_context_matrix)
                self.fold_2_lemmatizer['a'].load_preTrained_context_embeddings(self.cluster_context_matrix)
                self.fold_2_lemmatizer['b'].load_preTrained_context_embeddings(self.cluster_context_matrix)

            ## E-step
            self.Estep()
            ## M-step
            self.Mstep()
            ## Evaluate the model in its present state
            self.evaluate(print_statement='\nEVALUATING PURITY AFTER {} ROUNDS OF EM\n'.format(str(self.round+1)))
            if self.tune_latent_IC:
                ## U-step = update latent IC variable (and potentially latent base representation)
                self.Ustep()

            stdout.write('\nMAPPING CHANGE RATE: {}\n'.format(str(round(self.map_change_rate, 5))))
            stdout.flush()

            if self.map_change_rate < self.estop or self.round >= self.max_rounds-1:
                self.stop = True

            ### Saving the round's model
            pkl.dump(self.base_2_responsibilities, open(os.path.join(self.MODEL_DIR, 'step_{}_responsibilities.pkl'.format(self.round)), 'wb' ))

    def partition_random(self, n=2):

        if self.separateBases:
            tups = list(self.base_2_wf_2_cluster)

        else:
            tups = []
            for base in self.base_2_wf_2_cluster:
                for wf in self.base_2_wf_2_cluster[base]:
                    for cluster in self.base_2_wf_2_cluster[base][wf]:
                        if self.latent_IC:
                            clTup = (cluster, self.base_2_IC[base])
                        else:
                            clTup = (cluster, None)
                        tups.append((base, wf, clTup))
            
        random.shuffle(tups)

        return [tups[i::n] for i in range(n)]

    def get_s2s_splits(self, As, Bs):

        test_a = dict(self.test_set)
        test_b = dict(self.test_set)

        train_a = []
        train_b = []

        if self.separateBases:
            # go through all possible (base, wf), cluster combinations
            for tup in self.test_set:
                # if the base is an a_base and we think the cluster is right, we'll train on it
                (base, wf, clTup) = tup
                cluster = clTup[0]
                if base in As:
                    # if we train on it, the loss at test time will be biased
                    del test_a[tup]
                    if cluster in self.base_2_wf_2_cluster[base][wf]:
                        train_a.append(tup)
                # otherwise, if it's a b base, do the same
                else:
                    assert base in Bs
                    del test_b[tup]
                    if cluster in self.base_2_wf_2_cluster[base][wf]:
                        train_b.append(tup)

        else:
            # go through all possible (base, wf), cluster combinations
            last_was_b = True
            for tup in self.test_set:
                # if the base is an a_base and we think the cluster is right, we'll train on it
                if tup in As:
                    train_a.append(tup)
                    # if we train on it, the loss at test time will be biased
                    del test_a[tup]
                # otherwise, if it's a b base, do the same
                elif tup in Bs:
                    train_b.append(tup)
                    del test_b[tup]
                # make sure we're not making testing longer than it has to be
                else:
                    if last_was_b:
                        del test_a[tup]
                        last_was_b = False 
                    else:
                        del test_b[tup]
                        last_was_b = True
                
        # dev set samples from the other expert
        dev_a = list(random.sample(train_b, min(len(train_b), self.max_s2s_dev_size)))
        dev_b = list(random.sample(train_a, min(len(train_a), self.max_s2s_dev_size)))
        test_a = list(test_a)
        test_b = list(test_b)

        return train_a, dev_a, test_a, train_b, dev_b, test_b

    def Estep(self):

        stdout.write('\nE-STEP, ROUND {}\n'.format(self.round))
        stdout.flush()

        ### initialize s2s models
        self.fold_2_s2s = {'a': Inflector(args, os.path.join(os.path.join(self.MODEL_DIR, 'Inflection'), 'Round_{}_a'.format(self.round))), 'b': Inflector(args, os.path.join(os.path.join(self.MODEL_DIR, 'Inflection'), 'Round_{}_b'.format(self.round)))}
        self.fold_2_lemmatizer = {'a': Inflector(args, os.path.join(os.path.join(self.MODEL_DIR, 'Lemmatization'), 'Round_{}_a'.format(self.round))), 'b': Inflector(args, os.path.join(os.path.join(self.MODEL_DIR, 'Lemmatization'), 'Round_{}_b'.format(self.round)))}

        ### inf out responsibility matrix
        self.inf_out_responsibilities()

        ### Partition training data and get splits
        [a_bases, b_bases] = self.partition_random()
        train_a, dev_a, test_a, train_b, dev_b, test_b = self.get_s2s_splits(a_bases, b_bases)
        part_a = ('a', train_a, dev_a, test_a)
        part_b = ('b', train_b, dev_b, test_b)
        part_list = [part_a, part_b]

        self.train_partitions(part_list)

        self.record_responsibilities()

        greed = self.greed * (((self.dev_acc['a']/100) + (self.dev_acc['b']/100)) / 2)
        greedy_keep = 1 - greed
        stdout.write('\tGreedily pruning {}% of the possible analyses\n'.format(str(greed*100)))

        ### Convert losses to positive responsibilities
        for base in self.base_2_responsibilities:

            ## constant magnitudes
            self.base_2_responsibilities[base][0][np.isinf(self.base_2_responsibilities[base][0])] = -np.inf
            max_fin_loss = np.amax(self.base_2_responsibilities[base][0])
            self.base_2_responsibilities[base][0] -= max_fin_loss
            self.base_2_responsibilities[base][0][np.isneginf(self.base_2_responsibilities[base][0])] = np.inf
            self.base_2_responsibilities[base][0] *= -1

            ## warp magnitudes -> doesn't seem to have an effect either way
            # self.base_2_responsibilities[base][0] = 1/self.base_2_responsibilities[base][0]

            ## Regularize entropy across cluster columns -> seems to hurt performance
            # self.base_2_responsibilities[base][0] = softmax(self.base_2_responsibilities[base][0], axis=0)

            ### Greedily remove bad analyses
            for row_id in self.base_2_row_2_wf[base]:
                wf = self.base_2_row_2_wf[base][row_id]
                need = max(min(2, self.k), math.ceil(greedy_keep*len(self.still_eligible[(base, wf)])))
                subvec = []
                for cluster in self.still_eligible[(base, wf)]:
                    subvec.append((self.base_2_responsibilities[base][0][row_id][cluster] ,cluster))
                subvec.sort()
                for tup in subvec[:-need]:
                    cluster = tup[1]
                    clusTup = [cluster]
                    del self.still_eligible[(base, wf)][cluster]
                    if self.latent_IC:
                        IC = self.base_2_IC[base]
                        clusTup.append(IC)
                    else:
                        clusTup.append(None)
                    if self.model_context:
                        clusTup.append(cluster)
                    else:
                        clusTup.append(None)
                    fullTup = (base, wf, tuple(clusTup))
                    assert fullTup in self.test_set
                    del self.test_set[fullTup]

    def train_partitions(self, part_list):

        if self.parallel_partitions:
            with mp.Pool(2) as p:
                base_wf_tags_2_loss_list = p.map(self.train_partition, part_list, chunksize=1)
        else:
            base_wf_tags_2_loss_list = [self.train_partition(part_list[0]), self.train_partition(part_list[1])]
        self.partition_2_base_wf_tags_2_loss = {}
        self.partition_2_base_wf_tags_2_loss['a'] =  base_wf_tags_2_loss_list[0]
        self.partition_2_base_wf_tags_2_loss['b'] =  base_wf_tags_2_loss_list[1]
        self.dev_acc = {}
        self.dev_acc['a'] = self.fold_2_s2s['a'].trained_model.dev_acc
        self.dev_acc['b'] =  self.fold_2_s2s['b'].trained_model.dev_acc

    def train_partition(self, part):

        (partition, train_part, dev_part, test_part) = part

        if self.retrain_from_scratch or self.round > 0:
            stdout.write('\tS2S_{}: {}\n'.format(partition, self.fold_2_s2s[partition].work_dir))
            stdout.flush()
            self.fold_2_s2s[partition].prepare_s2s(train_part, dev=dev_part, test=test_part)
        else:
            raise NotImplementedError
            # stdout.write('\tUpdating S2S_{}: {}\n'.format(partition, self.fold_2_s2s[partition].model))
            # stdout.flush()
            # self.fold_2_s2s[partition].continue_s2s(train_part, dev=dev_part, test=test_part)

        base_wf_tags_2_loss = self.fold_2_s2s[partition].train_validate_s2s()

        return base_wf_tags_2_loss

    def record_responsibilities(self):

        stdout.write('\tCalculating responsibilities for all base,wf tuples\n')
        stdout.flush()
        ### Record responsibilities
        for tup in self.partition_2_base_wf_tags_2_loss['a']:
            (base, wf, (clusTup)) = tup 
            cluster = int(clusTup[0].split('_')[-1])
            wfi = self.base_2_responsibilities[base][1][wf]
            assert tup not in self.partition_2_base_wf_tags_2_loss['b']
            self.base_2_responsibilities[base][0][wfi][cluster] = self.partition_2_base_wf_tags_2_loss['a'][tup]
        for tup in self.partition_2_base_wf_tags_2_loss['b']:
            (base, wf, (clusTup)) = tup 
            cluster = int(clusTup[0].split('_')[-1])
            wfi = self.base_2_responsibilities[base][1][wf]
            assert tup not in self.partition_2_base_wf_tags_2_loss['a']
            self.base_2_responsibilities[base][0][wfi][cluster] = self.partition_2_base_wf_tags_2_loss['b'][tup]

    def inf_out_responsibilities(self):

        for base in self.base_2_responsibilities:
            ### shrink length if k has shrunk
            self.base_2_responsibilities[base][0] = np.array([[np.inf]*self.k]*len(self.base_2_responsibilities[base][1]))

    def Mstep(self):

        stdout.write('\nM-STEP, ROUND {}\n'.format(self.round))
        stdout.flush()

        self.tune_penalties()

        self.map_scores.append(self.map_score)

        if self.best_map_score == None:
            self.best_map_score = self.map_score
        else:
            improvement = self.map_score - self.best_map_score
            if improvement > 0.0:
                self.best_map_score = self.map_score
                # stdout.write('\tImproved over best map score by {}\n'.format(str(round(improvement, 3))))
            # else:
                # stdout.write('\tWorse then the best map score by {}\n'.format(str(round(improvement*-1, 3))))

    def get_plausability_score(self, G, cluster_2_realization_prob):

        clusters_realized = {}
        max_all = 0.0
        for row_id in G.assignments:
            for col_id in G.assignments[row_id]:
                max_all = max(max_all, G.original[row_id][col_id])
                if col_id not in clusters_realized:
                    clusters_realized[col_id] = {}
                clusters_realized[col_id][row_id] = True

        plausible_map_score = 1.0
        for col_id in range(self.k):
            P_doesnt_realize = (1-(cluster_2_realization_prob[col_id]))**len(clusters_realized)
            if col_id in clusters_realized:
                P_realized = 1-P_doesnt_realize
                for row_id in clusters_realized[col_id]: ## punishes over-abundance here
                    plausible_map_score *= (P_realized*G.original[row_id][col_id])
            else:
                max_r = np.amax(G.original_trans[col_id])
                plausible_map_score *= (P_doesnt_realize*(max_all-max_r)) ## if the best choice was good, punish for not choosing it

        return plausible_map_score

    def tune_penalties(self):

        ### PERFORM AN INITIAL MAPPING USING THE BEST PENALTY WEIGHTS FROM LAST ROUND
        cluster_2_realization_prob = {}
        base_2_clusters_realized = {}
        base_2_best_map = {}
        for c in range(self.k):
            cluster_2_realization_prob[c] = 0
        for base in self.base_2_row_2_wf:
            base_2_clusters_realized[base] = {}

            # Do bipartite mapping on the paradigm's responsibility matrix
            G = bipartite_match(self.base_2_responsibilities[base][0], row_penalty=self.base_2_penalties[base][0], column_penalty=self.base_2_penalties[base][1], assign_method=self.assign_method)
            base_2_best_map[base] = G
            for row_id in G.assignments:
                for col_id in G.assignments[row_id]:
                    cluster_2_realization_prob[col_id] += 1
                    if col_id not in base_2_clusters_realized[base]:
                        base_2_clusters_realized[base][col_id] = {}
                    base_2_clusters_realized[base][col_id][row_id] = True
        all_realizations = sum(list(cluster_2_realization_prob.values()))
        for cluster in cluster_2_realization_prob:
            cluster_2_realization_prob[cluster] /= all_realizations

        if self.tunable_penalties:

            ### FOR EACH PARADIGM, TUNE THE SYNCRETISM WEIGHTS (TUNE OA IN FUTURE WORK)
                # (COLUMN SPECIFIC SYNCRETISM PENTALTY IN FUTURE WORK)
            base_2_plausible_map_score = {}
            changes = 0
            no_changes = 0

            for base in self.base_2_row_2_wf:
                base_2_plausible_map_score[base] = self.get_plausability_score(base_2_best_map[base], cluster_2_realization_prob)
                no_improvement = True

                if self.debug:
                    stderr.write('\n\n_____\n')
                    stderr.write('{}\n'.format(base))
                    stderr.write('OLD MAP WITH SYNCRETISM PENALTY {}\n'.format(str(self.base_2_penalties[base][0])))
                    for row_id in base_2_best_map[base].assignments:
                        wf = self.base_2_row_2_wf[base][row_id]
                        for cluster in base_2_best_map[base].assignments[row_id]:
                            stderr.write('\t{} -> {}\n'.format(wf, cluster))
                    # stderr.write('Score: {}\n'.format(str(round(base_2_plausible_map_score[base], 4))))

                orig_syncretism_penalty = self.base_2_penalties[base][0]
                new_syncretism_penalty = self.base_2_penalties[base][0]
                new_score = base_2_plausible_map_score[base]

                while new_score >= base_2_plausible_map_score[base] and new_syncretism_penalty > 0.1:
                    new_syncretism_penalty -= 0.1
                    G = bipartite_match(self.base_2_responsibilities[base][0], row_penalty=new_syncretism_penalty, column_penalty=self.base_2_penalties[base][1], assign_method=self.assign_method)
                    new_score = self.get_plausability_score(G, cluster_2_realization_prob)

                    if new_score > base_2_plausible_map_score[base]:                            

                        self.base_2_penalties[base][0] = new_syncretism_penalty
                        base_2_best_map[base] = G
                        base_2_plausible_map_score[base] = new_score
                        no_improvement = False

                        if self.debug:
                            stderr.write('IMPROVED MAPPING!\n')
                            stderr.write('\nNEW MAP WITH SYNCRETISM PENALTY {}\n'.format(str(self.base_2_penalties[base][0])))
                            for row_id in base_2_best_map[base].assignments:
                                wf = self.base_2_row_2_wf[base][row_id]
                                for c_id in base_2_best_map[base].assignments[row_id]:
                                    stderr.write('\t{} -> {}\n'.format(wf, c_id))
                            stderr.write('Score: {}\n'.format(str(round(base_2_plausible_map_score[base], 4))))

                new_syncretism_penalty = orig_syncretism_penalty
                new_score = base_2_plausible_map_score[base]

                while new_score >= base_2_plausible_map_score[base] and new_syncretism_penalty < 0.9:
                    new_syncretism_penalty += 0.1
                    G = bipartite_match(self.base_2_responsibilities[base][0], row_penalty=new_syncretism_penalty, column_penalty=self.base_2_penalties[base][1], assign_method=self.assign_method)
                    new_score = self.get_plausability_score(G, cluster_2_realization_prob)

                    if new_score > base_2_plausible_map_score[base]:                            

                        self.base_2_penalties[base][0] = new_syncretism_penalty
                        base_2_best_map[base] = G
                        base_2_plausible_map_score[base] = new_score
                        no_improvement = False

                        if self.debug:
                            stderr.write('IMPROVED MAPPING!\n')
                            stderr.write('\nNEW MAP WITH SYNCRETISM PENALTY {}\n'.format(str(self.base_2_penalties[base][0])))
                            for row_id in base_2_best_map[base].assignments:
                                wf = self.base_2_row_2_wf[base][row_id]
                                for c_id in base_2_best_map[base].assignments[row_id]:
                                    stderr.write('\t{} -> {}\n'.format(wf, c_id))
                            stderr.write('Score: {}\n'.format(str(round(base_2_plausible_map_score[base], 4))))
            
                if no_improvement:
                    no_changes += 1
                else:
                    changes += 1

            stdout.write('{} MAPPINGS CHANGED AND {} MAPPINGS DIDNT CHANGE FROM TUNING PENALTIES\n'.format(changes, no_changes))

        ### UPDATE EVERYTHING ELSE GIVEN UPDATED WEIGHTS AND MAPPINGS

        self.wf_2_cluster = {}
        self.cluster_2_wf = {}
        self.cluster_2_base_2_wf = {}
        for i in range(self.k):
            self.cluster_2_wf[i] = {}
            self.cluster_2_base_2_wf[i] = {}
        last_base_2_wf_cluster = dict(self.base_2_wf_2_cluster)
        self.base_2_wf_2_cluster = {}
        self.map_score = 0.0
        total_rows = 0
        syncretisms = 0
        overabundances = 0

        for base in self.base_2_row_2_wf:
            row_id = -1
            self.base_2_wf_2_cluster[base] = {}

            # hard assignment forms to clusters/mixture components
            cache = {}
            for row_id in self.base_2_row_2_wf[base]:
                total_rows += 1
                wf = self.base_2_row_2_wf[base][row_id]
                self.base_2_wf_2_cluster[base][wf] = {}
                if wf not in self.wf_2_cluster:
                    self.wf_2_cluster[wf] = {}
                syncretisms -= 1
                for cluster in base_2_best_map[base].assignments[row_id].keys():
                    if cluster in cache:
                        overabundances += 1
                    else:
                        cache[cluster] = True
                    syncretisms += 1
                    self.base_2_wf_2_cluster[base][wf][cluster] = base_2_best_map[base].assignments[row_id][cluster]
                    self.wf_2_cluster[wf][cluster] = True 
                    self.cluster_2_wf[cluster][wf] = True
                    if base not in self.cluster_2_base_2_wf[cluster]:
                        self.cluster_2_base_2_wf[cluster][base] = {}
                    self.cluster_2_base_2_wf[cluster][base][wf] = True
                    if cluster not in self.still_eligible[(base, wf)]:
                        self.still_eligible[(base, wf)][cluster] = True 
                        clusTup = [cluster, None, None]
                        if self.latent_IC:
                            clusTup[1] = self.base_2_IC[base]
                        if self.model_context:
                            clusTup[2] = cluster
                        self.test_set[(base, wf, tuple(clusTup))] = True 


        stdout.write('\tMapped words: {}\n\tSyncretisms predicted: {}\n\tOver-abundance predicted: {}\n'.format(total_rows, syncretisms, overabundances))
        # stdout.write('\tMapped words: {}\n\tSyncretisms predicted: {}\n\tOver-abundance predicted: {}\n\tScore: {}\n'.format(total_rows, syncretisms, overabundances, str(round(self.map_score, 3))))
        stdout.flush()

        self.get_map_change_rate(last_base_2_wf_cluster)

        # if self.reconsider_cluster_count:
        # self.prune_obsolete_clusters()

    def get_map_change_rate(self, last_base_2_wf_cluster):

        match = last = new = 0.0
        # WILL NEED TO MAP OLD TO NEW BASES ONCE THE BASES BECOME LATENT VARIABLES
        for base in last_base_2_wf_cluster:
            for wf in last_base_2_wf_cluster[base]:
                last += len(last_base_2_wf_cluster[base][wf])
                new += len(self.base_2_wf_2_cluster[base][wf])
                match += len(set(last_base_2_wf_cluster[base][wf]).intersection(set(self.base_2_wf_2_cluster[base][wf])))
        
        if match == 0.0:
            self.map_change_rate = 1.0
        else:
            p = match / new 
            r = match / last
            f = 2* ((p*r)/(p+r))
            self.map_change_rate = 1 - f

    def prune_obsolete_clusters(self):

        ### CHECK IF OBSELETE CLUSTERS SHOULD BE PRUNED
        for cluster in range(self.k-1, -1, -1):
            if len(self.cluster_2_wf[cluster]) == 0:
                stdout.write('\nGETTING RID OF CLUSTER {} WHICH WAS NOT BEING USED\n'.format(cluster))
                self.k -= 1
                max_cluster = self.k
                ### MAP HIGHEST K TO THE CLUSTER GETTING DELETED IN ALL DATA STRUCTURES
                if max_cluster != cluster:
                    self.cluster_2_wf[cluster] = dict(self.cluster_2_wf[max_cluster])
                    self.cluster_2_base_2_wf[cluster] = dict(self.cluster_2_base_2_wf[max_cluster])
                    for base in self.cluster_2_base_2_wf[cluster]:
                        for wf in self.cluster_2_base_2_wf[cluster][base]:
                            self.wf_2_cluster[wf][cluster] = True 
                            del self.wf_2_cluster[wf][max_cluster]
                            self.base_2_wf_2_cluster[base][wf][cluster] = True 
                            del self.base_2_wf_2_cluster[base][wf][max_cluster]
                        self.base_2_penalties[base][0][cluster] = self.base_2_penalties[base][0][max_cluster]
                del self.cluster_2_wf[max_cluster]
                del self.cluster_2_base_2_wf[max_cluster]
                for base in self.base_2_penalties:
                    self.base_2_penalties[base][0] = self.base_2_penalties[base][0][:-1]
                toDel = []
                for (base, wf, tup) in self.test_set:
                    if tup[0] == max_cluster:
                        del self.still_eligible[(base, wf)][tup[0]]
                        toDel.append((base, wf, tup))
                for fullTup in toDel:
                    del self.test_set[fullTup]

    def evaluate(self, print_statement=None, eval_fn='purity'):

        if eval_fn == 'extrinsic':

            self.round = 99

            self.prepare_extrinsic_eval(self.extrinsic_inflection)
            mdl = self.extrinsic_inflection.train_validate_s2s_extrinsic()
            self.extrinsic_dev_acc, self.extrinsic_test_acc = mdl.dev_acc, mdl.test_acc

        elif eval_fn == 'benchmark':

            self.round = 99

            ### INITIALIZATION BASELINE
            self.prepare_extrinsic_eval(self.benchmark_initialization)
            mdl = self.benchmark_initialization.train_validate_s2s_extrinsic()
            init_dev_acc, init_test_acc = mdl.dev_acc, mdl.test_acc

            ### RANDOM BASELINE
            self.prepare_extrinsic_eval(self.benchmark_random, mode='random')
            mdl = self.benchmark_random.train_validate_s2s_extrinsic()
            random_dev_acc, random_test_acc = mdl.dev_acc, mdl.test_acc

            ### ORACLE BASELINE
            self.prepare_extrinsic_eval(self.benchmark_oracle, mode='oracle')
            mdl = self.benchmark_oracle.train_validate_s2s_extrinsic()
            oracle_dev_acc, oracle_test_acc = mdl.dev_acc, mdl.test_acc

            stdout.write('\n\nBENCHMARKS\n')
            stdout.write('\nRandom\nDev Acc: {} \t Test Acc: {}\n'.format(str(round(random_dev_acc, 4)), str(round(random_test_acc, 4))))

            stdout.write('\nInitialization\nDev Acc: {} \t Test Acc: {}\n'.format(str(round(init_dev_acc, 4)), str(round(init_test_acc, 4))))

            stdout.write('\nOracle (i.e., unfactored training)\nDev Acc: {} \t Test Acc: {}\n\n'.format(str(round(oracle_dev_acc, 4)), str(round(oracle_test_acc, 4))))

            exit()


        ## Purity evaluation
        elif eval_fn == 'purity':

            if self.debug:
                # print the base
                for base in self.base_2_responsibilities:
                    if self.latent_IC:
                        stderr.write('BASE: {}\n\tLatent IC: {}\n'.format(base, str(self.base_2_IC[base])))
                    else:
                        stderr.write('BASE: {}\n'.format(base))

                    # print each inflected form with its vector
                    stderr.write('\nRESPONSIBILITIES\n')
                    for row_id in self.base_2_row_2_wf[base]:
                        stderr.write('{} \t{}\n'.format(self.base_2_row_2_wf[base][row_id], str(self.base_2_responsibilities[base][0][row_id])))
                    stderr.write('*-*-*\n')
                    # print each predicted mapping
                    for row_id in self.base_2_row_2_wf[base]:
                        stderr.write('\n')
                        wf = self.base_2_row_2_wf[base][row_id]
                        for cluster in self.base_2_wf_2_cluster[base][wf]:
                            assert cluster in self.wf_2_cluster[wf]
                            assert wf in self.cluster_2_wf[cluster]
                            dynamic_weight = self.base_2_wf_2_cluster[base][wf][cluster]
                            stderr.write('PRED: {} -> Cluster: {}    (weight: {})\n'.format(wf, cluster, str(round(dynamic_weight, 4))))
                        # p1 = '*-*-* All {} clusters: {} ||| '.format(wf, ' | '.join(list(str(x) for x in self.wf_2_cluster[wf])))
                        
                        # print each gold mapping
                        if base in self.D.lem_2_cell_2_wf:
                            for cell in self.D.wf_2_cell[wf]:
                                if cell in self.D.lem_2_cell_2_wf[base] and wf in self.D.lem_2_cell_2_wf[base][cell]:
                                    stderr.write('GOLD: {} -> {}\n'.format(wf, cell))
                        else:
                            for cell in self.D.wf_2_cell[wf]:
                                stderr.write('GOLD: {} -> {}\n'.format(wf, cell))
                        # stderr.write('{}All {} cells: {}\n'.format(p1, wf, ' | '.join(list(self.D.wf_2_cell[wf]))))
                    stderr.write('\n___\n')

            self.purity = purityEval(self.cluster_2_wf, self.wf_2_cluster, self.D.cell_2_wf, self.D.wf_2_cell, msg=print_statement)

        else:
            raise NotImplementedError

    def Ustep(self):

        stdout.write('\nU-STEP, ROUND {}\n'.format(self.round))
        stdout.flush()

        ### Partition training data and get splits
        [a_bases, b_bases] = self.partition_random()
        train_a, dev_a, test_a, train_b, dev_b, test_b = self.get_lemmatization_splits(a_bases, b_bases)
        part_a = ('a', train_a, dev_a, test_a)
        part_b = ('b', train_b, dev_b, test_b)
        part_list = [part_a, part_b]

        self.train_lemmatization_partitions(part_list)

        ### RE-ASSIGN ICS FOR EACH PARADIGM BY ARGMIN AVERAGE LOSS ACROSS PARADIGM | IC
            ### COULD ALSO UPDATE BASES HERE TO BE WF SPECIFIC, BASED ON PREDICTIONS MADE WITH THAT IC
        self.reassign_ICs()

        if 'lemma' not in self.supervision:
            ## Update base predictions if we are not given the lemma as supervision
            stderr.write('NEED TO IMPLEMENT AUTOENCODER TO PREDICT NEW BASES!!!\n\tfor now we just keep the base as whatever the initialization predicted\n')
            # Need to update test set + all other structures containing bases

    def get_lemmatization_splits(self, As, Bs):

        train_a, train_b, dev_a, dev_b, test_a, test_b = [], [], [], [], [], []

        all_ICs = set(self.base_2_IC.values())

        for base in self.base_2_wf_2_cluster:
            ## consider sampling here!!
            for wf in self.base_2_wf_2_cluster[base]:
                for cluster in self.base_2_wf_2_cluster[base][wf]:
                    if base in self.undecided_baseIC: # or self.round % 5 == 0:
                        eligible_ICs = all_ICs
                    else:
                        eligible_ICs = [self.base_2_IC[base]]
                    for IC in eligible_ICs:
                        tup = [cluster, IC, None]
                        if self.model_context:
                            tup[2] = cluster
                        tup = (wf, base, tuple(tup))
                        if IC == self.base_2_IC[base]:
                            if base in As:
                                train_a.append(tup)
                            else:
                                train_b.append(tup)
                        if base in As:
                            test_b.append(tup)
                        else:
                            test_a.append(tup)
                
        # dev set samples from the other expert
        dev_a = list(random.sample(train_b, min(len(train_b), self.max_s2s_dev_size)))
        dev_b = list(random.sample(train_a, min(len(train_a), self.max_s2s_dev_size)))

        return train_a, dev_a, test_a, train_b, dev_b, test_b
        
    def train_lemmatization_partitions(self, part_list):

        if self.parallel_partitions:
            with mp.Pool(2) as p:
                wf_base_tags_2_loss_list = p.map(self.train_lemmatization_partition, part_list, chunksize=1)
        else:
            wf_base_tags_2_loss_list = [self.train_lemmatization_partition(part_list[0]), self.train_lemmatization_partition(part_list[1])]
        self.partition_2_wf_base_tags_2_loss = {}
        self.partition_2_wf_base_tags_2_loss['a'] =  wf_base_tags_2_loss_list[0]
        self.partition_2_wf_base_tags_2_loss['b'] =  wf_base_tags_2_loss_list[1]
        self.lem_dev_acc = {}
        self.lem_dev_acc['a'] = self.fold_2_lemmatizer['a'].trained_model.dev_acc
        self.lem_dev_acc['b'] =  self.fold_2_lemmatizer['b'].trained_model.dev_acc

    def train_lemmatization_partition(self, part):

        (partition, train_part, dev_part, test_part) = part

        if self.retrain_from_scratch or self.round > 0:
            stdout.write('\tLemmatizer_{}: {}\n'.format(partition, self.fold_2_lemmatizer[partition].work_dir))
            stdout.flush()
            self.fold_2_lemmatizer[partition].prepare_s2s(train_part, dev=dev_part, test=test_part)
        else:
            stdout.write('\tUpdating Lemmatizer_{}: {}\n'.format(partition, self.fold_2_lemmatizer[partition].model))
            stdout.flush()
            self.fold_2_lemmatizer[partition].continue_s2s(train_part, dev=dev_part, test=test_part)

        wf_base_tags_2_loss = self.fold_2_lemmatizer[partition].train_validate_s2s()
        stdout.write('\tLemmatizer_{} dev acc: {}\n'.format(partition, self.fold_2_lemmatizer[partition].trained_model.dev_acc))
        stdout.flush()

        return wf_base_tags_2_loss

    def reassign_ICs(self):

        stdout.write('\tReassigning IC for each base\n')
        stdout.flush()
        self.base_2_IC_2_avgLoss = {}

        for tup in self.partition_2_wf_base_tags_2_loss['a']:
            (wf, base, innerTup) = tup
            cluster = innerTup[0]
            IC = int(innerTup[1].split('_')[-1])
            assert tup not in self.partition_2_wf_base_tags_2_loss['b']
            if base not in self.base_2_IC_2_avgLoss:
                self.base_2_IC_2_avgLoss[base] = {}
            if IC not in self.base_2_IC_2_avgLoss[base]:
                self.base_2_IC_2_avgLoss[base][IC] = []
            self.base_2_IC_2_avgLoss[base][IC].append(self.partition_2_wf_base_tags_2_loss['a'][tup])

        for tup in self.partition_2_wf_base_tags_2_loss['b']:
            (wf, base, innerTup) = tup
            cluster = innerTup[0]
            IC = int(innerTup[1].split('_')[-1])
            assert tup not in self.partition_2_wf_base_tags_2_loss['a']
            if base not in self.base_2_IC_2_avgLoss:
                self.base_2_IC_2_avgLoss[base] = {}
            if IC not in self.base_2_IC_2_avgLoss[base]:
                self.base_2_IC_2_avgLoss[base][IC] = []
            self.base_2_IC_2_avgLoss[base][IC].append(self.partition_2_wf_base_tags_2_loss['b'][tup])

        orig_IC_count = len(set(self.base_2_IC.values()))

        for base in self.base_2_IC_2_avgLoss:
            if base in self.undecided_baseIC:
                min_loss = None
                for IC in self.base_2_IC_2_avgLoss[base]:
                    self.base_2_IC_2_avgLoss[base][IC] = sum(self.base_2_IC_2_avgLoss[base][IC])/len(self.base_2_IC_2_avgLoss[base][IC])
                    if min_loss == None:
                        min_loss = self.base_2_IC_2_avgLoss[base][IC]
                        best_IC = IC 
                    elif self.base_2_IC_2_avgLoss[base][IC] < min_loss:
                        best_IC = IC

                ### Update IC.. which essentially prunes the unused ones for next round
                if best_IC != self.base_2_IC[base]:
                    self.base_2_IC_consistency[base] = 0
                    self.base_2_IC[base] = best_IC
                    for tup in self.test_set:
                        (base2, wf, innerTup) = tup
                        IC = innerTup[1]
                        if base2 == base:
                            innerTup = list(innerTup)
                            innerTup[1] = best_IC
                            newTup = (base2, wf, tuple(innerTup))
                            del self.test_set[tup]
                            self.test_set[newTup] = True
            else:
                self.base_2_IC_consistency[base] += 1
                if self.base_2_IC_consistency[base] > self.IC_patience:
                    del self.undecided_baseIC[base]

        new_IC_count = len(set(self.base_2_IC.values()))

        if new_IC_count < orig_IC_count:
            stdout.write('\nREDUCED INFLECTION CLASSES FROM {} TO {}\n'.format(orig_IC_count, new_IC_count))
        elif new_IC_count == orig_IC_count:
            stdout.write('\nKEEPING ALL {} INFLECTION CLASSES\n'.format(new_IC_count))
        else:
            raise Exception('It shouldnt be possible to add inflection classes')

    def prepare_extrinsic_eval(self, mdl, mode='normal'): # normal, random, oracle

        stdout.write('\nBEGINNING EXTRINSIC EVALUATION\n\nMapping clusters to cell MSPSs\n')
        stdout.flush()
        cells = list(self.D.cell_2_wf)
        ## sanity check
        if len(cells) != self.k:
            stdout.write('\nWARNING! LEN CELLS ({}) != LEN CLUSTERS ({})!\n'.format(len(cells), len(self.k)))
        if len(self.D.wf_2_cell) != len(self.wf_2_cluster):
            stdout.write('\nWARNING! LEN GOLD WORDS ({}) != LEN CLUSTERED WORDS ({})!\n'.format(len(self.D.wf_2_cell), len(self.wf_2_cluster)))
        ##
        cells.sort()

        # weights = np.array([[0.0]*len(cells)]*self.k)

        # for cluster in range(self.k):
        #     for cell_id in range(len(cells)):
        #         weights[cluster][cell_id] = len(set(list(self.D.cell_2_wf[cells[cell_id]])).intersection(set(list(self.cluster_2_wf[cluster]))))
        
        # G = bipartite_match(weights, row_penalty=1.0, column_penalty=1.0, assign_method='hungarian')
        # self.cell_2_cluster = {}
        # for cluster in G.assignments:
        #     for cell_id in G.assignments[cluster]:
        #         cell = cells[cell_id]
        #         if cell in self.cell_2_cluster:
        #             stderr.write('WARNING! CELL {} MAPPED TO MULTIPLE CLUSTERS: {} & {}\n'.format(str(cell), str(self.cell_2_cluster[cell]), str(cluster)))
        #         self.cell_2_cluster[cell] = cluster

        self.cell_2_cluster = {}
        if mode == 'normal':
            for cell in self.D.cell_2_wf:
                max_score = -1
                for cluster in range(self.k):
                    score = len(set(list(self.D.cell_2_wf[cell])).intersection(set(list(self.cluster_2_wf[cluster]))))
                    if score > max_score:
                        best_cluster = cluster 
                        max_score = score
                self.cell_2_cluster[cell] = best_cluster
        else:
            cluster = -1
            for cell in self.D.cell_2_wf:
                cluster += 1
                if cluster < self.k:
                    self.cell_2_cluster[cell] = cluster 
                else:
                    self.cell_2_cluster[cell] = random.choice(list(range(self.k-1)))

        if self.debug and mode != 'random':
            o = open(os.path.join(self.MODEL_DIR, 'map_{}.txt'.format(mode)), 'w')
            for cell in self.cell_2_cluster:
                cluster = self.cell_2_cluster[cell]
                o.write('factored cell {} -> cluster {}\n'.format(cell, cluster))
            o.close()
        stderr.flush()


        ## Getting splits
        # train
        train, dev, test = [], [], []
        # for base in self.base_2_wf_2_cluster:
        #     for wf in self.base_2_wf_2_cluster[base]:
        #         for cluster in self.base_2_wf_2_cluster[base][wf]:
        #             innerTup = [cluster, None, None]
        #             # if self.latent_IC:
        #             #     innerTup[1] = self.base_2_IC[base]
        #             if self.model_context:
        #                 innerTup[2] = cluster
        #             innerTup = tuple(innerTup)
        #             train.append((base, wf, innerTup))
        # shared task train
        (lem_2_cell_2_wf, cell_2_wf, wf_2_cell) = self.D.gold_train_sup

        for lem in lem_2_cell_2_wf:
            base = lem
            if 'lemma' not in self.supervision:
                if lem in self.lem_2_base:
                    base = self.lem_2_base[lem]
                else:
                    base = 'OOV'
            wf_cells = {}
            for cell in lem_2_cell_2_wf[lem]:
                for wf in lem_2_cell_2_wf[lem][cell]:
                    wf_cells[(wf, cell)] = True
            for (wf, cell) in wf_cells:

                if mode == 'normal':
                    innerTups = []
                    for cluster in self.base_2_wf_2_cluster[base][wf]:
                        innerTup = [cluster, None, None]
                        if self.latent_IC:
                            innerTup[1] = self.base_2_IC[base]
                        if self.model_context:
                            innerTup[2] = cluster
                        innerTups.append(tuple(innerTup))
                elif mode == 'random':
                    cluster = random.choice(list(range(self.k-1)))
                    innerTup = [cluster, None, None]
                    if self.latent_IC:
                        innerTup[1] = self.base_2_IC[base]
                    if self.model_context:
                        innerTup[2] = cluster
                    innerTup = tuple(innerTup)
                elif mode == 'oracle':
                    cluster = self.cell_2_cluster[cell]
                    innerTup = [cluster, None, None]
                    if self.latent_IC:
                        innerTup[1] = self.base_2_IC[base]
                    if self.model_context:
                        innerTup[2] = cluster
                    innerTup = tuple(innerTup)
                if mode == 'normal':
                    for innerTup in innerTups:
                        train.append((base, wf, innerTup))
                else:
                    train.append((base, wf, innerTup))

        # dev
        (lem_2_cell_2_wf, cell_2_wf, wf_2_cell) = self.D.gold_dev
        for lem in lem_2_cell_2_wf:
            base = lem
            if 'lemma' not in self.supervision:
                if lem in self.lem_2_base:
                    base = self.lem_2_base[lem]
                else:
                    base = 'OOV'
            for cell in lem_2_cell_2_wf[lem]:
                for wf in lem_2_cell_2_wf[lem][cell]:
                    cluster = self.cell_2_cluster[cell]
                    innerTup = [cluster, None, None]
                    if self.latent_IC:
                        innerTup[1] = -1
                        if base in self.base_2_IC:
                            innerTup[1] = self.base_2_IC[base]
                    if self.model_context:
                        innerTup[2] = cluster
                    innerTup = tuple(innerTup)
                    dev.append((base, wf, innerTup))
        # test
        (lem_2_cell_2_wf, cell_2_wf, wf_2_cell) = self.D.gold_test

        for lem in lem_2_cell_2_wf:
            base = lem
            if 'lemma' not in self.supervision:
                if lem in self.lem_2_base:
                    base = self.lem_2_base[lem]
                else:
                    base = 'OOV'
            for cell in lem_2_cell_2_wf[lem]:
                for wf in lem_2_cell_2_wf[lem][cell]:
                    cluster = self.cell_2_cluster[cell]
                    innerTup = [cluster, None, None]
                    if self.latent_IC:
                        innerTup[1] = -1
                        if base in self.base_2_IC:
                            innerTup[1] = self.base_2_IC[base]
                    if self.model_context:
                        innerTup[2] = cluster
                    innerTup = tuple(innerTup)
                    test.append((base, wf, innerTup))

        # get context vectors
        if self.model_context:
            self.wf_2_context_embedding, self.exponent_2_context_embedding, self.cluster_2_context_embedding, full_context_matrix, wf_cluster_or_exp_2_embedding_row = get_context_vectors(self.base_2_wf_2_cluster, self.cluster_2_base_2_wf, self.context_instances, self.context_vocab)

            self.cluster_context_matrix = []
            for cluster in range(self.k):
                self.cluster_context_matrix.append(self.cluster_2_context_embedding[cluster])
            self.cluster_context_matrix = np.array(self.cluster_context_matrix)

            mdl.load_preTrained_context_embeddings(self.cluster_context_matrix)

        stdout.write('\nBuilding {} extrinsic inflection model: {}\n'.format(mode, mdl.model))
        stdout.flush()
        mdl.prepare_s2s(train, dev=dev, test=test)
        

####################################################################################
### Functions
####################################################################################

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def purityEval(cluster_2_wf, wf_2_cluster, cell_2_wf, wf_2_cell, warn=True, msg=None):

    if msg != None:
        stdout.write('\n{}\n'.format(msg))
        stdout.flush()

    if warn:
        try:
            assert len(wf_2_cluster) == len(wf_2_cell)
        except AssertionError:
            stderr.write('WARNING!\nTHE NUMBER OF FORMS CLUSTERED ({}) DOES NOT MATCH THE NUMBER OF GOLD FORMS ({})!\n\n'.format(len(wf_2_cluster), len(wf_2_cell)))

    denominator = 0
    numerator = 0
    for cluster in cluster_2_wf:
        cluster_mates = set(cluster_2_wf[cluster])
        denominator += len(cluster_mates)
        max_numerator = 0
        for cell in cell_2_wf:
            cell_mates = set(cell_2_wf[cell])
            max_numerator = max(max_numerator, len(cluster_mates.intersection(cell_mates)))
        numerator += max_numerator

    purity = 100*numerator/denominator

    if msg == None:
        stderr.write('Purity: {}%\nClusters: {}\nActual Cells: {}\nClustered Forms: {}\nActual Forms: {}\n'.format(str(round(purity, 2)), str(len(cluster_2_wf)), str(len(cell_2_wf)), str(len(wf_2_cluster)), str(len(wf_2_cell))))
    else:
        stdout.write('Purity: {}%\nClusters: {}\nActual Cells: {}\nClustered Forms: {}\nActual Forms: {}\n'.format(str(round(purity, 2)), str(len(cluster_2_wf)), str(len(cell_2_wf)), str(len(wf_2_cluster)), str(len(wf_2_cell))))
        stdout.flush()

    return purity

def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--benchmark_only', default=False, type=str2bool, help='If True, we skip EM and move to the extrinsic evaluation, where we try 3 techniques for mapping latent cluster variables to ground truth cells before inflecting: (1) random assignment, (2) using the initialization to get assignments, (3) oracularly replacing the ground truth cells with unique integers, which is comparable to the unfactored training setup discussed in Kann & Schutze, 2016.')
    parser.add_argument('--eval_extrinsic', default=True, type=str2bool, help='If True, use shared task data for an extrinsic evaluation.')

    ################################################################################
    ### LOCATION AND TYPE OF DATA SETS: UD, UNIMORPH, OR SIGMORPHON SHARED TASK
    ################################################################################
    parser.add_argument('-s', '--STdir', type=str, help='Location of the directory containing Shared Task files for training and testing. We expect these files to be formatted after the SIGMORPHON 2018 Task 1 convention, 3 tab separated columns for lemma/base, inflected form, and morpho-syntactic property set with filenames [language]-split(-trainSize) as in https://github.com/sigmorphon/conll2018/tree/master/task1/all/.', default=None)
    parser.add_argument('-u', '--UMdir', type=str, help='Location of the directory containing UniMorph subdirectories. Each subdirectory should be named for the language it covers, containing a file of the same name. These files should be formatted after the standard convention, 3 tab separated columns for lemma/base, inflected form, and morpho-syntactic property set.', default=None)
    parser.add_argument('-d', '--UDdir', type=str, help='Location of the directory containing all Universal Dependency files for training and testing. We expect this directory to contain files formatted after the... THIS REQUIRES MORE WORK BECAUSE WE HAVE TO CONVERT TO UNIMORPH REPRESENTATION.', default=None)
    parser.add_argument('-r', '--raw_corpus', type=str, help='Text corpus to be used for modeling context and/or filtering data with realistic attestations. The corpus should already be lowercased and tokenized if your training/testing data expects such input - we do not perform any additional preprocessing.', default=None)
    parser.add_argument('-m', '--model_location', type=str, help='Directory where the model outputs will be stored.', default='MyModel/model')

    ################################################################################
    ### CONSTRAINTS: LEVEL OF SUPERVISION, TRAIN SIZE, LANGUAGE, POS
    ################################################################################
    parser.add_argument('--filter_data', default=True, type=str2bool, help='If True, filter UniMorph data by what is attested in the supplied raw corpus and what is not attested in the supplied shared task dev or test sets.')
    parser.add_argument('-l', '--language', type=str, help='Language must match how it is written in the UM and/or UD directories.', required=True)
    parser.add_argument('-p', '--POS', type=str, choices=['all', 'N', 'V', 'ADJ'], help='This constrains what parts-of-speeches we train and test on.', default='all')
    parser.add_argument('-c', '--constrained_tag_set', nargs='+', help='If this argument is specified, we will limit training and evaluation to only forms realizing the exact cells specified.', default=None)
    parser.add_argument('-t', '--train_size', type=str, choices=['high', 'medium', 'low'], help='Since this task is unsupervised, it is not unreasonable to see the full data during training, i.e., train-high + dev + test, of course, without access to cell information. But we can also train on progressively smaller subsets, i.e., train-high, medium, or low, to see how well the mixture model handles progressively smaller partially attested paradigms and progressively fewer overall training instances.', default='high')
    parser.add_argument('-S', '--supervision', nargs='+', help='What information is available to aid in inflection?.', choices=['paradigm', 'lemma', 'cell'], default=['paradigm', 'lemma'])
    parser.add_argument('--max_train_paradigms', type=int, help='Maximum number of paradigms which will be extracted from the training data.', default=None)
    parser.add_argument('--max_train_instances', type=int, help='Maximum number of unique lem,wf tuples which will be extracted from the training data.', default=None)

    ################################################################################
    ### BIPARTITE MATCHING PARAMETERS
    ################################################################################
    parser.add_argument('-a', '--assign_method', type=str, help='Method of bipartite matching forms to cells/clusters.', choices=['kilimanjaro', 'hungarian'], default='kilimanjaro')
    parser.add_argument('-o', '--overabundance_penalty', type=float, help='Used by the Kilimanjaro bipartite mapping function to punish overabundance. This parameter is tunable, though this can be used to set it. Using values of 1 for this and the syncretism penalty results in the same behavior as the Hungarian algorithm given a square matrix of graph edges. Lower values allow for more permissive treatment of syncretism or overabundance', default=1.0)
    parser.add_argument('-y', '--syncretism_penalty', type=float, help='Used by the Kilimanjaro bipartite mapping function to punish overabundance. See description for overabundance_penalty.', default=1.0)
    parser.add_argument('-k', '--k', type=str, help='If integer, set number of clusters to propose to k. If "blind", take the max number of forms per paradigm. If "oracle", take the gold number of cells per paradigm.', default='blind')
    parser.add_argument('-D', '--embed_dim', type=int, help='Dimension of the embedded form representation.', default=100)  # Must be the same as d_model?

    ################################################################################
    ## ANALYZER PARAMETERS
    ################################################################################
    parser.add_argument('--ana_greed', type=float, help='Fraction of possible cluster analyses to discard after each step, will be multiplied by average devloss so that greed increases up to this value as the model converges.', default=0.5)
    parser.add_argument('--ana_rand_init', default=False, type=str2bool, help='If True, word-cluster mappings will be initialized randomly.')
    parser.add_argument('--ana_tunable_penalties', default=False, type=str2bool, help='If True, re-tune syncretism penalty weights by paradigm during each M step.')
    parser.add_argument('--ana_latent_IC', default=True, type=str2bool, help='If True, we will model inflection class membership with a discrete latent variable.')
    parser.add_argument('--ana_tune_latent_IC', default=True, type=str2bool, help='If True, we will tune the discrete latent inflection class variable with a lemmatization model trained after each M-step.')
    parser.add_argument('--ana_retrain_from_scratch', default=True, type=str2bool, help='If False, we will update the previous s2s model given new training data at each E-step. Otherwise, we will re-train the s2s model from scratch on the new data.')
    parser.add_argument('-R', '--ana_max_rounds', type=int, help='Number of rounds after which, EM will be terminated, regardless of convergence.', default=20)
    parser.add_argument('--ana_IC_patience', type=int, help='Number of rounds after which, if the IC class has not changed for a base, it will become fixed.', default=3)
    parser.add_argument('--ana_estop', type=float, help='The EM algorithm will terminate once the proportion of mapping changes falls below this threshold.', default=5e-3)
    parser.add_argument('--ana_debug', default=False, type=str2bool, help='Prints responsibility matrices after each M-step.')
    parser.add_argument('--ana_parallel_partitions', default=False, type=str2bool, help='Trains both partitioned inflection models in parallel. May not work on GPU where the individual models are already running in parallel.')

    ################################################################################
    ## COMPONENT SEQ 2 SEQ MODEL PARAMETERS
    ################################################################################
    parser.add_argument('--s2s_max_dev_size', default=200, type=int, help='maximum number of instances to calculate dev loss on.')
    parser.add_argument('--s2s_data_format', default=dataloader.DataFormat.INFLECTION, type=dataloader.DataFormat,
        # choices=['INFLECTION', 'INFLECTION_IC', MT'],
        help='Data files should contain one instance per line. This flag defines the '
          'syntax of instances. The following options are supported: '
          'MT: 2 tab delimited fields. (1) space-delimited source sentence (2) '
          'space-delimited target sentence. '
          'INFLECTION: 3 tab deimited fields. (1) lemma as undelimited characters '
          '(2) target as undelimeted characters (3) morphosyntactic property set '
          'describing (2). This is the standard UniMorph format. '
          'INFLECTION_IC: 4 tab deimited fields. Like INFLECTION but with an extra field for inflection class.'
          ### ADD ADDITIONAL FORMATS FOR HANDLING CONTEXT EMBEDDINGS.
          )
    parser.add_argument('--s2s_model', default=model_lib.ModelFormat.TRANSFORMER, type=model_lib.ModelFormat,
    # choices=['TRANSFORMER'],
    help='Only one architecture is supported at present: '
      'TRANSFORMER: Soft, non-monotonic multi-head attention without recurrent '
      'cells. Input is assumed to be one-dimensional with features treated as '
      'additional sequence elements, as in the KANN_2016 set up.'
      )
    # Defining model hyperparameters.
    parser.add_argument('--s2s_max_num_epochs', default=40, type=int, help='Maximum number of epochs.')
    parser.add_argument('--s2s_patience', default=10, type=int,
    help='The number of epochs the accuracy on the '
    'dev set is allowed not to improve before training stops.')
    parser.add_argument('--s2s_batch_size', default=64, type=int, help='Size of batches fed to the model.')
    parser.add_argument('--s2s_val_batch_size', default=1000, type=int, help='Size of batches during validation.')
    parser.add_argument('--s2s_optimizer', default='adam', type=str, choices=['adam', 'adadelta'], help='Optimization algorithm.')
    parser.add_argument('--s2s_epsilon', default=0.000000001, type=float, help='Small constant for stability.')
    # Hyperparameters specific to transformer model.
    # Notation is kept consistent with this implementation:
    # https://www.tensorflow.org/beta/tutorials/text/transformer
    parser.add_argument('--s2s_num_layers', default=4, type=int, help='Number of encoder and decoder layers.')
    parser.add_argument('--s2s_d_model', default=128, type=int, help='Similar to num_units in the RNN models.')
    parser.add_argument('--s2s_num_heads', default=8, type=int, help='Number of attention heads.')
    parser.add_argument('--s2s_dff', default=512, type=int, help='Dimensions in the feed forward network in each individual encoder and decoder layer.')
    parser.add_argument('--s2s_dropout_rate', default=0.1, type=float, help='Dropout rate.')
    parser.add_argument('--s2s_beta_1', default=0.9, type=float, help='Used for learning rate annealing.')
    parser.add_argument('--s2s_beta_2', default=0.98, type=float, help='Used for learning rate annealing.')
    parser.add_argument('--s2s_warmup_steps', default=4000, type=int, help='Learning rate will increase until this number of steps have been reached, then the learning rate will start annealing.')

    ################################################################################
    ## CONTEXT MODELING MODEL PARAMETERS
    ################################################################################
    parser.add_argument('--exponent_vocab', default=True, type=str2bool, help='Models distribution of exponents in context.')
    parser.add_argument('--model_context', default=True, type=str2bool, help='If False, no distributional data will be leveraged.')

    args = parser.parse_args()
    if args.ana_latent_IC or args.s2s_data_format == dataloader.DataFormat.INFLECTION_IC:
        args.s2s_data_format = dataloader.DataFormat.INFLECTION_IC
        args.ana_latent_IC = True

    return args

####################################################################################
### Main
####################################################################################

if __name__ == '__main__':

    ####################################################################################
    ### PARSING INPUT DATA
    ####################################################################################
    args = get_arguments()
    # assert args.embed_dim == args.d_model  # Does this really need to be true?
    D = Data_manager(args)
    D.print_data_summary()

    ####################################################################################
    ### INITIALIZING ANALYZER
    ####################################################################################
    ANA = Analyzer(args, D)

    ####################################################################################
    ### HARD EXPECTATION MAXIMIZATION USING SEQ 2 SEQ MODELS
    ####################################################################################
    ANA.run_EM()

    ####################################################################################
    ### EVALUATE THE FINAL MODEL ON HELD OUT INFLECTION TASK DATA
    ####################################################################################
    if args.benchmark_only:
        ANA.evaluate(eval_fn='benchmark')
    elif ANA.eval_extrinsic:
        ANA.evaluate(eval_fn='extrinsic')
        stdout.write('\n      _\n     / \\ \n    /   \\    \n __/_____\\__\n   /     \\\n  | o   o |\n| |   v   | |\n| | \\___/ | |\n|  \\_____/  |\n \\____|____/\n      |\n      |\n      |\n      |\n      |\n     / \\\n    /   \\\
            \n   /     \\\n  |       |\n  |       |\n  |       |\nFinal Purity: {}\nExtrinsic Dev Inflection Accuracy: {}\nExtrinsic Test Inflection Accuracy: {}\n\n'.format(str(round(ANA.purity, 4)), str(round(ANA.extrinsic_dev_acc, 4)), str(round(ANA.extrinsic_test_acc, 4))))
    else:
        stdout.write('\n      _\n     / \\ \n    /   \\    \n __/_____\\__\n   /     \\\n  | o   o |\n| |   v   | |\n| | \\___/ | |\n|  \\_____/  |\n \\____|____/\n      |\n      |\n      |\n      |\n      |\n     / \\\n    /   \\\
            \n   /     \\\n  |       |\n  |       |\n  |       |\nFinal Purity: {}\n\n'.format(str(round(ANA.purity, 4))))

    ### Saving the final analyzer
    # pkl.dump(ANA, open(os.path.join(ANA.MODEL_DIR, 'FINAL_ANALYZER.pkl'), 'wb' ))

