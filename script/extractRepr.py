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
    import cPickle as pickle
else:
    import pickle

from Seq2seq import seq2seq_runner, dataloader
from Seq2seq import model as model_lib

from s2sFlags import *
from utils import edist_alt, edist

if __name__ == "__main__":
  args = get_arguments()
  run = args.run

  variant = 0
  workdir = "%s/model%d" % (run, variant)
  while os.path.exists(workdir):
      variant += 1
      workdir = "%s/model%d" % (run, variant)

  workdir = "%s/repr" % run
  os.system("rm -rf %s" % workdir)
  flags = S2SFlags(args, workdir)

  cpdir = os.path.abspath("%s/model%d/checkpoints/" % (run, variant - 1))
  cpt = None
  for fi in os.listdir(cpdir):
      if fi.endswith(".index"):
          cpt = cpdir + "/" + fi
  assert(cpt is not None)
  cpt.replace(".index", "")
  print("Checkpoint", cpt)
  flags.checkpoint_to_restore = cpt

  flags.train = "%s/train.txt" % run
  flags.dev = "%s/dev.txt" % run

  hparams, flags = seq2seq_runner.handle_preparation_flags(flags)

  # Prepare data.
  all_data = seq2seq_runner.prepare_data(flags, hparams)
  trg_language_index = all_data.trg_language_index
  trg_feature_index = all_data.trg_feature_index
  trg_max_len_seq = all_data.trg_max_len_seq
  trg_max_len_ft = all_data.trg_max_len_ft
  split_sizes = all_data.split_sizes

  model = model_lib.Model(hparams, all_data, flags)

  cpName = hparams.checkpoint_to_restore.replace("ckpt-", "manual_save").replace(".index", ".h5")
  model.transformer.load_weights(cpName)
  enc = model.transformer.encoder.embedding
  print(enc)

  wts = enc.get_weights()

  langD = {}
  for ind, ti in enumerate(trg_language_index.tokens):
      if ti.startswith("TRG_LANG_"):
          lang = ti.replace("TRG_LANG_", "")
          langD[lang] = ind

  print(wts[0].shape, "shape of wts")
  wMat = np.zeros((len(langD), wts[0].shape[1]))
  langLst = []
  for rowInd, (lang, ind) in enumerate(langD.items()):
      wMat[rowInd] = wts[0][ind, :]
      langLst.append(lang)

  outfn = "%s/languageEmbedding.dump" % run
  with open(outfn, "wb") as outfh:
      pickle.dump([wMat, langLst], outfh)
