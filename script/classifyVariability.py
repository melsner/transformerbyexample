from __future__ import division, print_function
import sys
from collections import defaultdict
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

from Seq2seq import seq2seq_runner, dataloader
from Seq2seq import model as model_lib

from s2sFlags import *

import tensorflow as tf
import tensorflow.keras as tkeras

def writeInstances(insts, fn):
    base = 0
    total = 0
    with open(fn, "w") as fh:
        for inst in insts:
            for (src, targ, pred, correct) in inst:
                features = re.findall("TRG_([A-Z]*_[a-z]*)", src)
                content = re.sub("TRG_([A-Z]*_[a-z]*)", "", src)
                total += 1
                if correct:
                    sign = "T"
                    base += 1
                else:
                    sign = "F"
                fh.write("%s\t%s\t%s\n" % (content, sign, ";".join(features)))

    maj = max(base / total, (total - base) / total)
    print("Maj:", base, "/", total, maj)

def readPreds(fh):
    correct = True

    for line in fh:
        line = line.strip()
        if not line:
            continue

        if line.startswith("*ERROR*"):
            correct = False
        elif line.startswith("SRC:"):
            src = "".join(line[len("SRC:"):].split())
            #src = src.replace("_", " ")
            #src = src.replace("TRG LANG ", "TRG_LANG_")
        elif line.startswith("TRG:"):
            trg = "".join(line[len("TRG:"):].split())
            trg = trg.replace("_", " ")
        elif line.startswith("PRD:"):
            prd = "".join(line[len("PRD:"):].split())
            #prd = prd.replace("_", " ")
            yield src, trg, prd, correct
            src, trg, prd, correct = (None, None, None, True)

def readCases(fh):
    cases = []
    currSrc, currTarg = (None, None)
    curr = []
    for (source, targ, pred, correct) in readPreds(fh):
        sourceLemma = source.split(":")[0]
        if sourceLemma != currSrc or targ != currTarg:
            if curr:
                cases.append(curr)
            curr = []
            currSrc = sourceLemma
            currTarg = targ

        curr.append((source, targ, pred, correct))

    if curr:
        cases.append(curr)

    return cases

def caseVaries(insts):
    if len(insts) == 0:
        return False

    if np.all([correct is True for (src, targ, pred, correct) in insts]):
        return False
    if np.all([correct is False for (src, targ, pred, correct) in insts]):
        return False

    return True

TRAIN_STEP_SIGNATURE = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, 1), dtype=tf.float64),
]

def runModel(flags):
  tcls = buildModel(flags)
  tcls.train()
  #tcls.transformer.save_weights(flags.work_dir + "/classifier-model.h5")
  #print("Transformer layers", tcls.transformer.layers)
  #tcls.transformer.load_weights(flags.work_dir + "/classifier-model.h5")

def buildModel(flags):
  hparams, flags = seq2seq_runner.handle_preparation_flags(flags)

  # Prepare data.
  all_data = seq2seq_runner.prepare_data(flags, hparams)
  trg_language_index = all_data.trg_language_index
  trg_feature_index = all_data.trg_feature_index
  trg_max_len_seq = all_data.trg_max_len_seq
  trg_max_len_ft = all_data.trg_max_len_ft
  split_sizes = all_data.split_sizes

  # Get model.
  model = model_lib.Model(hparams, all_data, flags)
  cpName = hparams.checkpoint_to_restore.replace("ckpt-", "manual_save").replace(".index", ".h5")
  model.transformer.load_weights(cpName)

  transEnc = model.transformer.encoder
  dtrain = all_data.dataset_train
  tcls = TModel(hparams, all_data, flags, transEnc, dtrain)
  print("Built model")
  return tcls

class TModel(model_lib.Model):
    def __init__(self, hparams, all_data, flags, enc, dtrain):
        super(TModel, self).__init__(hparams, all_data, flags, shell=True)
        self.transformer = TransformerCls(enc)
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name="train_accuracy")

        self.checkpoint = tf.train.Checkpoint(transformer=self.transformer,
                                              optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.checkpoint, self.hparams.checkpoint_dir, max_to_keep=2)


    def train(self, fake=False):
        print("Running", self.hparams.max_num_epochs, "of classifier training")
        best_dev_acc = -1.0
        for epoch in range(self.hparams.max_num_epochs):

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (_, (inp, trg)) in enumerate(self.dataset_train):

                trg_conv = trg[:, 1]
                trg_txt = self.trg_language_index.decode(trg_conv)
                trg_vals = [xx == "T" for xx in trg_txt.split()]
                trg_vals = np.array(trg_vals, dtype="float64")[:, None]

                self.train_step(inp, trg_vals)

                if fake:
                    return

            dev_acc = self.validate()
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

    def validate(self):
        val_srcs, val_trgs = self.dev_srcs, self.dev_trgs
        reprs = self.prepare_for_forced_validation(val_srcs, self.src_language_index)
        enc_padding_mask = model_lib.create_padding_mask(reprs)
        prs = self.transformer.call(reprs, False, enc_padding_mask)
        prs = prs.numpy()[:, 0]
        decs = (prs > .5).astype("int64")
        acc = 0

        for ii in range(decs.shape[0]):
            trg_txt = val_trgs[ii]
            trg_val = (trg_txt == "T")

            if decs[ii] == trg_val:
                acc += 1

        acc /= decs.shape[0]

        return acc


    @tf.function(input_signature=TRAIN_STEP_SIGNATURE)
    def train_step(self, inp, trg):
        """Runs one batch of training as a graph-executable function."""

        enc_padding_mask = model_lib.create_padding_mask(inp)

        with tf.GradientTape() as tape:
            predictions = self.transformer(inp, True, enc_padding_mask)
            loss = self.loss_function(trg, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,
                                       self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(trg, predictions)

    def loss_function(self, real, pred):
        return self.loss_object(real, pred)

class TransformerCls(tf.keras.Model):
    def __init__(self, enc):
        super(TransformerCls, self).__init__()

        self.encoder = enc
        #this is a really bad idea
        #enc.trainable = False
        self.internalLayers = []
        for layer in range(3):
            self.internalLayers.append(tkeras.layers.Dense(128, activation="linear"))
            self.internalLayers.append(tkeras.layers.BatchNormalization())
            self.internalLayers.append(tkeras.layers.Activation("relu"))
        self.dOut = tkeras.layers.Dense(1, activation="sigmoid")

    def call(self, inp, training, enc_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        #d1Out = self.d1(enc_output)
        output = enc_output
        for li in self.internalLayers:
            output = li(output)
        final = self.dOut(output)
        return final[:, 0, :]

if __name__ == "__main__":
    print("Script start")
    args = get_arguments()
    print("Arguments parsed")
    run = args.run
    dfile = args.data

    with open(dfile) as dfh:
        cases = readCases(dfh)

    variableCases = []
    for inst in cases:
        if caseVaries(inst):
            variableCases.append(inst)
            #for cx in inst:
            #    print(cx)
            #print()

    print("Variability rate:", len(variableCases), "/", len(cases), ":", 
          len(variableCases) / len(cases))

    assert(args.load_other is not None)
    if not args.load_other.endswith(".index"):
        print("Searching for latest checkpoint")
        best = 0
        bestC = None
        lst = os.listdir(args.load_other)

        for cpt in lst:
            if cpt.endswith(".index"):
                cptN = int(cpt.replace(".index", "").split("-")[1])
                if cptN > best:
                    best = cptN
                    bestC = cpt

        assert(bestC is not None)
        args.load_other += "/" + bestC
        print("Using", args.load_other)

    variant = 0
    workdir = os.path.dirname(args.load_other) + "/../../classify-variable%d" % variant
    while os.path.exists(workdir):
        variant += 1
        workdir = os.path.dirname(args.load_other) + "/../../classify-variable%d" % variant

    os.makedirs(workdir, exist_ok=True)

    np.random.shuffle(variableCases)
    nDev = 200
    dev = variableCases[:nDev]
    train = variableCases[nDev:]
    writeInstances(dev, workdir + "/dev.txt")
    writeInstances(train, workdir + "/train.txt")

    os.system("rm -rf %s/model" % workdir)

    flags = S2SFlags(args, workdir + "/model")
    flags.train = workdir + "/train.txt"
    flags.dev = workdir + "/dev.txt"
    flags.checkpoint_to_restore = os.path.abspath(args.load_other)

    print("Starting run")
    runModel(flags)
    #seq2seq_runner.run(flags)
