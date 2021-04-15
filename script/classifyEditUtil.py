from __future__ import division, print_function
import sys
from collections import defaultdict, Counter
import os
import re
import numpy as np
import argparse
import random
import math
import six
if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl

import tensorflow as tf
import tensorflow.keras as tkeras
import tensorflow_datasets as tfds

from Seq2seq import seq2seq_runner
from Seq2seq import model as model_lib
from Seq2seq import dataloader as dld

from s2sFlags import *

class Vocab:
  def __init__(self, tokens):
    self.alphaToInd = {}
    self.indToAlpha = {}
    self.nChars = 0

    for ti in sorted(tokens):
        self.get(ti)
  
  def get(self, ss, closed=False):
    val = self.alphaToInd.get(ss, None)
    if val is not None:
      return val

    if closed:
        return None

    self.alphaToInd[ss] = self.nChars
    self.indToAlpha[self.nChars] = ss
    self.nChars += 1
    return self.nChars

def readData(hparams, flags):
    lemmaTokens = set(["<PAD>", "<UNK>"])
    cells = set()
    rules = defaultdict(set)
    instances = []
    maxLen = 0

    with open(flags.train) as ifh:
        for line in ifh:
            (lemma, cell, rule) = line.strip().split("\t")
            lemma = lemma.replace(" ", "_")
            lemma = ["<BOS>"] + list(lemma) + ["<EOS>"]
            lemma = tuple(lemma)
            instances.append((lemma, cell, rule))
            
            lemmaTokens.update(lemma)
            cells.add(cell)
            rules[cell].add(rule)
            maxLen = max(len(lemma), maxLen)

    devInstances = []
    with open(flags.dev) as ifh:
        for line in ifh:
            (lemma, cell, rule) = line.strip().split("\t")
            lemma = lemma.replace(" ", "_")
            lemma = ["<BOS>"] + list(lemma) + ["<EOS>"]
            lemma = tuple(lemma)
            devInstances.append((lemma, cell, rule))

    lemmaIndex = Vocab(lemmaTokens)
    cellIndex = Vocab(cells)
    ruleIndex = {}
    for ci, ruleSet in rules.items():
        ruleIndex[ci] = Vocab(ruleSet)

    writeMapFile(flags.work_dir, lemmaIndex, cellIndex, ruleIndex, maxlen=maxLen)

    train = ClassificationData(instances, lemmaIndex, cellIndex, ruleIndex, rules, maxLen, hparams.batch_size, truncate=True)
    dev = ClassificationData(devInstances, lemmaIndex, cellIndex, ruleIndex, rules, maxLen, hparams.batch_size)
    return train, dev

def writeMapFile(wdir, lemmaIndex, cellIndex, ruleIndex, maxlen):
    with open("%s/vocab_map.tsv" % wdir, "w") as ofh:
        for ind, alpha in lemmaIndex.indToAlpha.items():
            ofh.write("%s\t%s\t%s\n" % ("src_vocab", ind, alpha))

        for ind, alpha in cellIndex.indToAlpha.items():
            ofh.write("%s\t%s\t%s\n" % ("cell_vocab", ind, alpha))

        for cell, voc in ruleIndex.items():
            for ind, alpha in voc.indToAlpha.items():
                ofh.write("%s\t%s\t%s\t%s\n" % ("rule_vocab", cell, ind, alpha))

        ofh.write("maxlen\t%d\n" % maxlen)

class ClassificationData(tkeras.utils.Sequence):
    def __init__(self, instances, lemmaIndex, cellIndex, ruleIndex, rules, maxLen, batchSize, truncate=False):
        super(ClassificationData, self).__init__()
        self.instances = instances
        self.lemmaIndex = lemmaIndex
        self.cellIndex = cellIndex
        self.ruleIndex = ruleIndex
        self.batchSize = batchSize
        self.maxLen = maxLen
        self.truncate = truncate

        self.cells = sorted(rules.keys())
        self.nOutcomes = {}
        self.rules = rules
        for ci, ruleSet in rules.items():
            self.nOutcomes[ci] = len(ruleSet)

    def shuffle(self):
        np.random.shuffle(self.instances)

    def __len__(self):
        if self.truncate:
            return len(self.instances) // self.batchSize
        else:
            return int(np.ceil(len(self.instances) / self.batchSize))

    def __getitem__(self, idx):
        #print("Getting batch", idx)
        bStart = idx * self.batchSize
        bSize = min(self.batchSize, len(self.instances) - bStart)
        #turn the lemmas into indices
        #turn the rule into the correct label vectors
        inds = np.zeros((bSize, self.maxLen))
        rules = [ np.zeros((bSize, self.nOutcomes[cell])) for cell in self.cells ]

        for exI, ii in enumerate(range(bStart, bStart + bSize)):
            lemma, cell, rule = self.instances[ii]
            for jj, ci in enumerate(lemma):
                char = self.lemmaIndex.get(ci, closed=True)
                if char is None:
                    char = self.lemmaIndex.get("<UNK>", closed=True)
                inds[exI, jj] = char

            for jj in range(len(lemma), self.maxLen):
                inds[exI, jj] = self.lemmaIndex.get("<PAD>", closed=True)

            cellInd = self.cellIndex.get(cell, closed=True)
            if cellInd is None:
                continue
            ruleInd = self.ruleIndex[cell].get(rule, closed=True)
            if ruleInd is None:
                continue
            
            rules[cellInd][exI, ruleInd] = 1

        return inds, rules

class MultiOutModel(object):
    def __init__(self, hparams, flags, enc, cells):
        self.hparams = hparams
        self.flags = flags
        self.transformer = TransformerMOut(enc, cells)
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        learning_rate = model_lib.CustomSchedule(self.hparams.d_model,
                                                 warmup_steps=self.hparams.warmup_steps)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=self.hparams.beta_1, beta_2=self.hparams.beta_2,
            epsilon=self.hparams.epsilon)

        self.checkpoint = tf.train.Checkpoint(transformer=self.transformer,
                                              optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.checkpoint, self.hparams.checkpoint_dir, max_to_keep=2)

    def train(self, data, devData, fake=False):
        if not fake:
            print("Running", self.hparams.max_num_epochs, "of classifier training")

        best_dev_acc = -1.0
        for epoch in range(self.hparams.max_num_epochs):

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (_, (inp, trg)) in enumerate(data):

                self.train_step(inp, trg)

                if fake:
                    return

            data.shuffle()
            dev_acc = self.validate(devData)
            sys.stderr.write(
                'Epoch {} Loss {:.4f} Train Acc {:.4f} Dev Acc {:.4f}\n'.format(
                    epoch + 1, self.train_loss.result(), self.train_accuracy.result(), dev_acc))

            if dev_acc > best_dev_acc:
                self.best_checkpoint_path = self.ckpt_manager.save()
                self.transformer.save_weights(self.hparams.checkpoint_dir + "/manual_save%d.h5" % self.checkpoint.save_counter)
                sys.stderr.write(
                    'Saving checkpoint for epoch {} at {}\n'.format(
                        epoch + 1, self.best_checkpoint_path))
                best_dev_acc = dev_acc

    @tf.function
    def train_step(self, inp, trg):
        """Runs one batch of training as a graph-executable function."""

        enc_padding_mask = model_lib.create_padding_mask(inp)

        with tf.GradientTape() as tape:
            predictions = self.transformer(inp, True, enc_padding_mask)
            loss = self.loss_function(trg, predictions)
            #print("Loss:", loss)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,
                                       self.transformer.trainable_variables))

        self.train_loss(loss)
        for (ri, pi) in zip(trg, predictions):
            self.train_accuracy(ri, pi)

    def loss_function(self, real, pred):
        losses = [self.loss_object(ri, pi) for (ri, pi) in zip(real, pred)]
        #print("******")
        #print(losses)
        losses = tf.stack(losses, axis=-1)
        result = tf.reduce_sum(losses)
        #print("-->", result)
        return result

    def validate(self, data):
        acc = 0
        total = 0

        for (_, (inp, trg)) in enumerate(data):
            enc_padding_mask = model_lib.create_padding_mask(inp)
            prs = self.transformer(inp, False, enc_padding_mask)
            total += trg[0].shape[0]
            for (ri, pi) in zip(trg, prs):
                pi = pi.numpy()
                rounded = (pi > .5).astype("int64")
                prod = (ri * pi)
                acc += np.sum(prod)

        return acc / total

    def wordReprs(self, data):
        res = []
        for (_, (inp, trg)) in enumerate(data):
            enc_padding_mask = model_lib.create_padding_mask(inp)
            reprs = self.transformer.representations(inp, enc_padding_mask)
            res.append(reprs.numpy())

        return np.vstack(res)

class TransformerMOut(tf.keras.Model):
    def __init__(self, enc, cells):
        super(TransformerMOut, self).__init__()

        self.encoder = enc
        self.internalLayers = []
        for layer in range(1):
            self.internalLayers.append(tkeras.layers.Dense(128, activation="linear"))
            self.internalLayers.append(tkeras.layers.BatchNormalization())
            self.internalLayers.append(tkeras.layers.Activation("relu"))

        self.dOuts = []
        for cell, dim in cells:
            self.dOuts.append(tkeras.layers.Dense(dim, activation="softmax"))

    def call(self, inp, training, enc_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        output = enc_output[:, 0, :]
        for li in self.internalLayers:
            output = li(output)
        finals = [di(output) for di in self.dOuts]
        return finals

    def representations(self, inp, enc_padding_mask):
        enc_output = self.encoder(inp, False, enc_padding_mask)
        output = enc_output[:, 0, :]
        for li in self.internalLayers:
            output = li(output)
        return output
