from __future__ import absolute_import, division, print_function

import os
import torch
import random
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def semeval_score():
    output = os.popen('bash test.sh')
    text = [i for i in output.read().split("\n") if i != ""]
    score = text[-2]
    words = score.split()
    prec, recall, f1 = float(words[2][:-1]), float(words[5][:-1]), float(words[8][:-1])
    return {"precision": prec, "recall": recall, "f1 score": f1}


def construct_nonull_res(rs, ks):
    r_rs, r_ks = [], []
    for r, k in zip(rs, ks):
        if r != 0:
            r_rs.append(r)
            r_ks.append(k)
    return r_rs, r_ks


def financial_acc_and_f1(preds, labels):
    recall = recall_score(labels, preds, average="micro")
    precision = precision_score(labels, preds, average="micro")
    f1 = f1_score(labels, preds, average='micro')
    p_r, p_k = construct_nonull_res(preds, labels)
    r_k, r_r = construct_nonull_res(labels, preds)
    nonull_recall = recall_score(r_k, r_r, average='micro')
    nonull_precision = precision_score(p_k, p_r, average='micro')
    return {"acc": nonull_precision,
            "recall": recall,
            "precision": precision,
            "f1_score": f1,
            "nonull_recall": nonull_recall,
            "nonull_precision": nonull_precision}


def compute_metrics(args, preds, labels):
    assert len(preds) == len(labels)
    if args.task_name == "semeval":
        return semeval_score()
    elif args.task_name == "financial":
        return financial_acc_and_f1(preds, labels)
    else:
        raise KeyError(args.task_name)


def set_seed(args, add_param=0):
    random.seed(args.seed + add_param)
    np.random.seed(args.seed + add_param)
    torch.manual_seed(args.seed + add_param)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed + add_param)
