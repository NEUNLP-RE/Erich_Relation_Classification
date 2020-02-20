from __future__ import absolute_import, division, print_function

import os


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
