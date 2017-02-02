import sys
import os
import logging
import numpy as np
import pandas as pd
from scipy import stats


program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

def calc_word_sim(model, eval_file):
    df = pd.read_csv(eval_file, sep=',', header=0) # eval dataset
    col1, col2, score = df.columns.values
    model_vocab = model.vocab.keys()
    ground = []
    sys = []
    for idx, row in df.iterrows():
        if row[col1] in model_vocab and row[col2] in model_vocab:
            ground.append(float(row[score]))
            sys.append(model.similarity(row[col1], row[col2]))

    # compute Spearman's rank correlation coefficient (https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
    print sys
    # import pdb;pdb.set_trace()
    corr, p_val = stats.spearmanr(sys, ground)
    logger.info("# of pairs found: %s / %s" % (len(ground), len(df)))
    logger.info("correlation: %s" % corr)
    return corr, p_val

