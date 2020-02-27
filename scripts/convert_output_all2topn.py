from __future__ import print_function

import argparse
import codecs


def add_parameters(description):
    params = argparse.ArgumentParser(description=description)
    params.add_argument("--input", "-i", required=True, help="the old output needed to convert")
    params.add_argument("--old_relation", "-o", required=True, help="the old relation(all relations)")
    params.add_argument("--new_relation", "-r", required=True, help="the new relation(part relations)")
    params.add_argument("--output", "-n", required=True, help="the new output")
    return params


def construct_i2r(f_name, i2r=True):
    k2v = {}
    with codecs.open(f_name, "r", "utf-8") as fr:
        for l in fr:
            ids, rel = l.strip().split()
            if not i2r:
                ids, rel = rel, ids
            k2v[ids] = rel
    return k2v


if __name__ == "__main__":
    parsers = add_parameters("convert old input with old relation to new output with new relation")
    args = parsers.parse_args()

    old_ids2relation = construct_i2r(args.old_relation, i2r=True)
    new_relation2ids = construct_i2r(args.new_relation, i2r=False)

    with codecs.open(args.input, "r", "utf-8") as fi, codecs.open(args.output, "w", "utf-8") as fo:
        for l in fi:
            blocks = l.strip().split()
            rels = blocks[0].split(",")
            new_ids = []
            for r in rels:
                old_rel = old_ids2relation[r]
                ids = new_relation2ids[old_rel] if new_relation2ids.get(old_rel) else new_relation2ids["NULL"]
                new_ids.append(ids)
            blocks[0] = ",".join(list(set(new_ids)))
            fo.write(" ".join(blocks) + "\n")

    print("End...")
