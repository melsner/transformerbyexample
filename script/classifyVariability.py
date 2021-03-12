from __future__ import division, print_function
import sys
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

from Seq2seq import seq2seq_runner, dataloader
from Seq2seq import model as model_lib

from s2sFlags import *

import tensorflow as tf
import tensorflow.keras as tkeras

TRAIN_STEP_SIGNATURE = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, 1), dtype=tf.float64),
]


def runModel(flags):
  tcls = buildModel(flags)
  tcls.train()
  tcls.transformer.save_weights(flags.work_dir + "/classifier-model.h5")
  print("Transformer layers", tcls.transformer.layers)
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

    def train(self, fake=False):
        print("Running", self.hparams.max_num_epochs, "of classifier training")
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

            #dev_acc = self.validate()
            sys.stderr.write(
                'Epoch {} Loss {:.4f} Train Acc {:.4f}\n'.format(
                    epoch + 1, self.train_loss.result(), self.train_accuracy.result()))

            # print("let's see some preds?")
            # for (ind, (inp, trg)) in enumerate(self.dataset_train):
            #     trg_conv = trg[:, 1]
            #     trg_txt = self.trg_language_index.decode(trg_conv)
            #     trg_vals = [xx == "T" for xx in trg_txt.split()]
            #     trg_vals = np.array(trg_vals, dtype="float64")[:, None]

            #     enc_padding_mask = model_lib.create_padding_mask(inp)                
            #     predictions = self.transformer(inp, True, enc_padding_mask)
            #     print(ind, predictions, trg_vals)
            #     print(self.loss_function(trg, predictions))
            #     if ind > 5:
            #         break

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
        self.d1 = tkeras.layers.Dense(128, activation="tanh")
        self.dOut = tkeras.layers.Dense(1, activation="sigmoid")

    def call(self, inp, training, enc_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        d1Out = self.d1(enc_output)
        final = self.dOut(d1Out)
        return final[:, 0, :]

if __name__ == "__main__":
    print("Script start")
    args = get_arguments()
    print("Arguments parsed")
    run = args.run
    dfile = args.data

    rawData = np.loadtxt(dfile, dtype=str, delimiter="\t")

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
        args.load_other += "/" + cpt
        print("Using", args.load_other)

    variant = 0
    workdir = os.path.dirname(args.load_other) + "/../../classify-variable%d" % variant
    while os.path.exists(workdir):
        variant += 1
        workdir = os.path.dirname(args.load_other) + "/../../classify-variable%d" % variant

    flags = S2SFlags(args, workdir)
    flags.train = dfile
    flags.dev = dfile
    flags.checkpoint_to_restore = os.path.abspath(args.load_other)

    print("Starting run")
    runModel(flags)
