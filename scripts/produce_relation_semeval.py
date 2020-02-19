from __future__ import print_function

import argparse
import codecs


def add_parameters(descriptions):
    params = argparse.ArgumentParser(description=descriptions)
    params.add_argument("--sem_file", "-s", required=True, help="source file of Semeval dataset")
    params.add_argument("--rel", "-r", required=True, help="relation2id")
    return params


if __name__ == "__main__":
    parsers = add_parameters("extract relations for semeval")
    args = parsers.parse_args()

    with codecs.open(args.sem_file, "r", "utf=8") as fs:
        lines = [l.strip() for l in fs.readlines()]

    relations = sorted(list(set(lines[1::4])))
    if "Other" in relations:
        # move "Other" to the end of list
        relations.remove("Other")
        relations.append("Other")

    idx = range(len(relations))
    rel2ids = dict(zip(relations, idx))
    with codecs.open(args.rel, "w", "utf-8") as fr:
        for r in relations:
            fr.write("\t".join([r, str(rel2ids[r])]) + "\n")
    print("End...")
