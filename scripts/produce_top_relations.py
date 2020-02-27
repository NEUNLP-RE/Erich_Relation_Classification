from __future__ import print_function

import argparse
import codecs

no_direction_relations = [
    "NULL",
    "金融关系/产品--产品关系/竞争",
    "金融关系/公司--人关系/合作",
    "金融关系/公司--公司关系/合作",
    "金融关系/公司--公司关系/竞争"
]
need_relations = [
    "金融关系/公司--人关系/投资",
    "金融关系/公司--产品关系/投资",
    "金融关系/公司--公司关系/投资",
    "金融关系/公司--人关系/起诉",
    "金融关系/公司--公司关系/起诉",
    "金融关系/公司--产品关系/收购",
    "金融关系/公司--人关系/收购",
    "金融关系/公司--公司关系/并购",
    "金融关系/公司--产品关系/生产"
]


def add_parameters(description):
    params = argparse.ArgumentParser(description=description)
    params.add_argument("--nodir_static", "-n", required=True, help="the no direction relation static")
    params.add_argument("--dir_static", "-d", required=True, help="the direction relation static")
    params.add_argument("--topn", "-t", type=int, default=40,
                        help="the top relations num")
    params.add_argument("--relation", "-r", required=True, help="the new direction relations")
    params.add_argument("--new_nodir_static", "-c", required=True, help="the new no direction static")
    params.add_argument("--new_dir_static", "-s", required=True, help="the new direction static")
    return params


def read_static(f_name):
    record = {}
    with codecs.open(f_name, "r", "utf-8") as fr:
        for l in fr:
            rel, num = l.strip().split()
            record[rel] = num
    return record


def init_dir_rels(rel_list):
    n_r = []
    for r in rel_list:
        if r in no_direction_relations:
            n_r.append(r)
        else:
            n_r += [r + "_l2r", r + "_r2l"]
    return n_r


if __name__ == "__main__":
    parsers = add_parameters(description="produce topn direction relations")
    args = parsers.parse_args()

    nodir_record = read_static(args.nodir_static)
    dir_record = read_static(args.dir_static)

    nodir_list = sorted(nodir_record.items(), key=lambda x: int(x[1]), reverse=True)
    nodir_num = len(nodir_list)
    new_dir_rels = init_dir_rels(need_relations)
    for rel, _ in nodir_list:
        if len(new_dir_rels) >= args.topn:
            break
        if rel in need_relations:
            continue
        new_dir_rels += init_dir_rels([rel])
    new_dir_rels = sorted(new_dir_rels)

    with codecs.open(args.relation, "w", "utf-8") as fr:
        new_drnum = len(new_dir_rels)
        for ind in range(new_drnum):
            fr.write(str(ind) + " " + new_dir_rels[ind] + "\n")

    new_nodir_rels = []
    for ndr in new_dir_rels:
        ndr_s = ndr.split("_")[0]
        if ndr_s not in new_nodir_rels:
            new_nodir_rels.append(ndr_s)

    null_num = 0
    for ind in range(nodir_num):
        if nodir_list[ind][0] == "NULL" or nodir_list[ind][0] not in new_nodir_rels:
            null_num += int(nodir_list[ind][1])

    with codecs.open(args.new_nodir_static, "w", "utf-8") as fnns:
        for rel, num in nodir_list:
            if rel == "NULL":
                num = str(null_num)
            if rel in new_nodir_rels:
                fnns.write(rel + "\t" + num + "\n")
    with codecs.open(args.new_dir_static, "w", "utf-8") as fnds:
        for ndr in new_dir_rels:
            num = dir_record[ndr]
            if ndr == "NULL":
                num = str(null_num)
            fnds.write(ndr + "\t" + num + "\n")

    print("End...")
