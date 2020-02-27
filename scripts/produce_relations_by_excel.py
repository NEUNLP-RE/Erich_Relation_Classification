from __future__ import print_function

import argparse
import codecs
import xlrd
import re

no_direction_relations = [
    "NULL",
    "金融关系/产品--产品关系/竞争",
    "金融关系/公司--人关系/合作",
    "金融关系/公司--公司关系/合作",
    "金融关系/公司--公司关系/竞争"
]


def add_parameters(description):
    params = argparse.ArgumentParser(description=description)
    params.add_argument("--excel", "-e", required=True, help="excel")
    params.add_argument("--relations", "-r", required=True, help="relation2id")
    return params


if __name__ == "__main__":
    parsers = add_parameters("convert source file of old relation to new one of new relation")
    args = parsers.parse_args()

    workbook = xlrd.open_workbook(args.excel)
    sheet = workbook.sheets()[0]
    l_num = sheet.nrows
    rel_list = []
    for i in range(1, l_num):
        sample = sheet.row_values(i)
        relations = re.split("[|,]", sample[3])
        for r in relations:
            if r == "" or r in rel_list:
                continue
            rel_list.append(r)

    relation_dir = []
    for r in rel_list:
        if r in no_direction_relations:
            relation_dir.append(r)
        else:
            relation_dir += [r + "_l2r", r + "_r2l"]
    relation_dir = sorted(relation_dir)

    ids2relation = []
    for i in range(len(relation_dir)):
        ids2relation.append([str(i), relation_dir[i]])

    with codecs.open(args.relations, "w", "utf-8") as fr:
        for i2r in ids2relation:
            fr.write(" ".join(i2r) + "\n")
