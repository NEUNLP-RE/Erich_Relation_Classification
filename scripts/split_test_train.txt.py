from __future__ import print_function

import argparse
import codecs
import random
import copy


train_null_limit = 1500
test_null_limit = 200
test_percent = 0.1


def add_parameters(description):
    params = argparse.ArgumentParser(description=description)
    params.add_argument("--source_file", "-s", required=True, help="the file need to split")
    params.add_argument("--relations", "-r", required=True, help="the relation file")
    params.add_argument("--test_file", "-c", required=True, help="the test file")
    params.add_argument("--train_file", "-x", required=True, help="the train file")
    return params


def construct_dataset(file):
    data = []
    with codecs.open(file, "r", "utf-8") as f:
        for l in f:
            ws = l.strip().split()
            data.append([ws[0], " ".join(ws[1:5]), ws[-1]])
    return data


def check_repeat_sent(md, sent):
    if len(md) >= 1 and sent == md[-1][0]:
        return len(md) - 1
    return -1


def merge_the_same_sentence(data):
    merged_data = []
    for d in data:
        pos = check_repeat_sent(merged_data, d[-1])
        if pos == -1:
            merged_data.append([d[-1], [d[:-1]]])
        else:
            merged_data[pos][1].append(d[:-1])
    return list(merged_data)


def legal_join(sample, c):
    all_classes = sum([rs.split(",") for rs in [r[0] for r in sample[1]]], [])
    reduce_flag = False
    for cla in all_classes:
        if c[cla] == -1:
            continue
        else:
            c[cla] = max(-1, c[cla] - 1)
            reduce_flag = True
    return reduce_flag


def extract_dataset(data):
    counter = copy.deepcopy(relation_sample)
    test, train = [], []
    train_null = 0
    for d in data:
        if legal_join(d, counter):
            test.append(d)
        else:
            null_num = sum([rs.split(",") for rs in [r[0] for r in d[1]]], []).count("0")
            train_null += null_num
            if null_num == 0 or train_null <= train_null_limit:
                train.append(d)

    return test, train


def init_relation_sample(rel_file):
    rel_s = {}
    with codecs.open(rel_file, "r", "utf-8") as fr:
        for l in fr:
            ids, r = l.strip().split()
            rel_s[ids] = 0
    return rel_s


def static_dataset(data, rel_s):
    for d in data:
        rels = d[0].split(",")
        for r in rels:
            rel_s[r] += 1
    for k in rel_s.keys():
        rel_s[k] = int(test_percent * rel_s[k])
    if rel_s["0"] > test_null_limit:
        rel_s["0"] = test_null_limit
    return rel_s


if __name__ == "__main__":
    parsers = add_parameters("split test and train file")
    args = parsers.parse_args()

    relation_sample = init_relation_sample(args.relations)
    dataset = construct_dataset(args.source_file)
    relation_sample = static_dataset(dataset, relation_sample)
    dataset = merge_the_same_sentence(dataset)
    random.shuffle(dataset)
    test_dataset, train_dataset = extract_dataset(dataset)
    random.shuffle(train_dataset)
    with codecs.open(args.test_file, "w", "utf-8") as ft:
        for td in test_dataset:
            for rel in td[1]:
                sample_for_write = rel + [td[0]]
                ft.write(" ".join(sample_for_write) + "\n")

    with codecs.open(args.train_file, "w", "utf-8") as ft:
        for td in train_dataset:
            for rel in td[1]:
                sample_for_write = rel + [td[0]]
                ft.write(" ".join(sample_for_write) + "\n")

    print("End...")
