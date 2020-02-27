from __future__ import print_function

import argparse
import xlrd
import re


def add_parameters(description):
    params = argparse.ArgumentParser(description=description)
    params.add_argument("--source_excel", "-e", required=True, help="the excel file needed to parse")
    params.add_argument("--relations", "-r", required=True, help="the relations 2 ids")
    params.add_argument("--output_file", "-o", required=True, help="the output file")
    return params


def filter_multi_entity_data(samp):
    """remove "" relation
    1. 实体关系之间有 "" 类型，去除这样的关系，重构单个样本
    2. 同一实体：相同实体没有关系
    :param samp: one sample in dataset
    :return:
    """
    es1, es2, rs = samp[1].split("|"), samp[2].split("|"), samp[3].split("|")
    if len(es1) != len(es2) != len(rs):
        return None
    sent_relations = []
    for e1, e2, r in zip(es1, es2, rs):
        if r != "" and e1 != e2 and e1 != "" and e2 != "":
            sent_relations.append([e1, e2, r])
    samp = [samp[0], sent_relations] if len(sent_relations) else None
    return samp


def filter_dataset(instance):
    """filter some illegal data
    1. 原始数据过滤字符，et. " ", "\xa0", "　"...
    2. 指定sample包含6个元素，并且确定是金融关系或者NULL
    3. 违法条件：
        (1) 残缺实体：2个实体有实体缺失
        (2) 同一实体：相同实体没有关系
    4. 去除多关系
        (1) 未知关系：实体关系之间有 "" 类型，去除这样的关系
        (2) 同一实体：相同实体没有关系
    :param instance: the sample of all dataset
    :return: legal_data, illegal_data
    """
    for index in range(len(instance)):
        instance[index] = re.sub("[ \xa0　]", "", instance[index])
    assert len(instance) == 4
    instance = filter_multi_entity_data(instance)
    return instance


def parser_sample(d):
    """将多实体多关系分割成多份sample"""
    es1, es2, rs = zip(*d[1])
    assert len(es1) == len(es2) == len(rs)
    return list(es1), list(es2), list(rs)


def produce_patten(e):
    e = e.replace(" ", "")
    new_e = ""
    for c in e:
        if c in ["(", ")", "+", "*", "?"]:
            new_e += "\\"
        new_e += c
    return new_e


def nearest_entity(es1_l, es2_l):
    dis = 10000
    d1, d2 = None, None
    for e1 in es1_l:
        for e2 in es2_l:
            if abs(sum(e1) - sum(e2)) < dis:
                d1, d2 = e1, e2
                dis = abs(sum(e1) - sum(e2))
    assert d1 and d2
    return d1, d2


def illegal_entity(e1_pos, e2_pos):
    return e2_pos[0] <= e1_pos[0] <= e2_pos[1] or e2_pos[0] <= e1_pos[1] <= e2_pos[1] \
           or e1_pos[0] <= e2_pos[0] <= e1_pos[1] or e1_pos[0] <= e2_pos[1] <= e1_pos[1]


def construct_label(r_tag, e1_pos, e2_pos, r_dict):
    relations = r_dict.keys()
    rels = r_tag.split(",")
    r_l = []
    for r in rels:
        if r in relations:
            r_l.append(r)
            continue
        if e1_pos[0] < e2_pos[0]:
            append_str = "_l2r"
        else:
            append_str = "_r2l"
        r_l.append(r + append_str)
    str_r = ",".join([str(r_dict[r_sample]) for r_sample in r_l])
    return str_r


def get_pos(sent):
    ent = re.search(r"{0}(.*?){0}".format("<en>"), sent)
    all_pos = [list(m.span()) for m in re.finditer(produce_patten(ent.group()), sent)]
    len_pos = len(all_pos)
    all_pos = [[all_pos[l_i][0] - 8*l_i, all_pos[l_i][1] - 8*(l_i + 1) - 1] for l_i in range(len_pos)]
    return all_pos


def construct_all_pos(text, entity):
    text = re.sub(produce_patten(entity), "<en>" + entity + "<en>", text)
    all_pos = get_pos(text)
    sample_t = re.sub("<en>" + produce_patten(entity) + "<en>", " " * len(entity), text)
    return all_pos, sample_t


def construct_all_samples(t, es1, es2, rs, r_dict):
    b = []
    for es1, es2, r in zip(es1, es2, rs):
        longer, shorter = sorted([es1, es2], key=lambda x: len(x), reverse=True)
        sample_t = t
        if sample_t.find(longer) == -1:  # 3498行数据
            break
        longer_pos, sample_t = construct_all_pos(sample_t, longer)
        if sample_t.find(shorter) == -1:
            break
        shorter_pos, sample_t = construct_all_pos(sample_t, shorter)
        all_e1_pos, all_e2_pos = (longer_pos, shorter_pos) if longer == es1 else (shorter_pos, longer_pos)
        assert all_e1_pos and all_e2_pos
        e1_pos, e2_pos = nearest_entity(all_e1_pos, all_e2_pos)
        assert not illegal_entity(e1_pos, e2_pos)
        label = construct_label(r, e1_pos, e2_pos, r_dict)
        if e1_pos[0] > e2_pos[0]:
            e1_pos, e2_pos = e2_pos, e1_pos
        b.append([label] + e1_pos + e2_pos + [t])
    return b


def get_id_dataset(database, relation_dict):
    all_d = []
    for d in database:
        text = d[0]
        entities1, entities2, relations = parser_sample(d)
        batch = construct_all_samples(text, entities1, entities2, relations, relation_dict)
        all_d += batch
    return all_d


if __name__ == "__main__":
    parsers = add_parameters(description="parse excel file to output file")
    args = parsers.parse_args()

    excel = xlrd.open_workbook(args.source_excel)
    table = excel.sheets()[0]
    # col_values = table.col_values(0)
    row_num = table.nrows
    keys = table.row_values(0)
    dataset, illegal = [], []
    for i in range(1, row_num):
        excel_sample = table.row_values(i)
        sample = filter_dataset(excel_sample)
        if sample:
            dataset.append(sample)
        else:
            illegal.append(excel_sample)
    print("filter all dataset: leave {0}, abandon {1}".format(len(dataset), len(illegal)))

    with open(args.relations, "r", encoding="utf-8") as f_r:
        relation2id = {}
        for k in f_r:
            ids, relation = k.strip().split()
            relation2id[relation] = ids

    train_dataset = get_id_dataset(dataset, relation2id)
    print("construct dataset: orig {0}, produce {1}".format(len(dataset), len(train_dataset)))

    with open(args.output_file, "w", encoding="utf-8") as fw:
        for td in train_dataset:
            fw.write(" ".join(["%s" % e for e in td]) + "\n")

    print("End...")
