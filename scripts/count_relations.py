from __future__ import print_function

import argparse
import codecs


def add_parameters(description):
    params = argparse.ArgumentParser(description=description)
    params.add_argument("--text", "-t", required=True, help="the count source file in format")
    params.add_argument("--relations", "-r", required=True, help="the relation file")
    params.add_argument("--static_dir", "-s", required=True, help="the result of statistics result")
    params.add_argument("--static_nodir", "-n", required=True, help="the result of no direction statistics result")
    return params


if __name__ == "__main__":
    parsers = add_parameters(description="count relations for text in format")
    args = parsers.parse_args()

    rel_record = {}
    ids2norel = {}
    norel_record = {}
    with codecs.open(args.relations, "r", "utf-8") as fr:
        for l in fr:
            ids, rel = l.strip().split()
            rel_record[ids] = [rel, 0]

            no_dir_rel = rel.split("_")[0]
            ids2norel[ids] = no_dir_rel
            if not norel_record.get(no_dir_rel, None):
                norel_record[no_dir_rel] = 0

    with codecs.open(args.text, "r", "utf-8") as ft:
        for l in ft:
            blocks = l.strip().split()
            rels = blocks[0].split(",")
            for r in rels:
                rel_record[r][1] += 1
                norel_record[ids2norel[r]] += 1

    with codecs.open(args.static_dir, "w", "utf-8") as fs:
        for i in rel_record.keys():
            rel, num = rel_record[i]
            fs.write("{}\t{}\n".format(rel, num))

    with codecs.open(args.static_nodir, "w", "utf-8") as fns:
        for k in norel_record.keys():
            num = norel_record[k]
            fns.write(k + "\t" + str(num) + "\n")
