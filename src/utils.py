from __future__ import absolute_import, division, print_function

import os
import torch
import random
import numpy as np


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def semeval_score():
    output = os.popen('bash test.sh')
    text = [i for i in output.read().split("\n") if i != ""]
    score = text[-2]
    words = score.split()
    prec, recall, f1 = float(words[2][:-1]), float(words[5][:-1]), float(words[8][:-1])
    return {"precision": prec, "recall": recall, "f1 score": f1}


def compute_metrics(args, preds, labels):
    assert len(preds) == len(labels)
    if args.task_name == "semeval":
        return semeval_score()
    else:
        raise KeyError(args.task_name)


def set_seed(args, add_param=0):
    random.seed(args.seed + add_param)
    np.random.seed(args.seed + add_param)
    torch.manual_seed(args.seed + add_param)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed + add_param)
