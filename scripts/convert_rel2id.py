from __future__ import print_function

import argparse
import codecs


def add_parameters(description):
    params = argparse.ArgumentParser(description=description)
    params.add_argument("--id_file", "-i", required=True, help="eval/test id file")
    params.add_argument("--rel", "-r", required=True, help="relation2id")
    params.add_argument("--test", "-t", required=True, help="eval/test relation file")
    return params


if __name__ == "__main__":
    parsers = add_parameters("convert id file to relation file")
    args = parsers.parse_args()

    id2rel = {}
    with codecs.open(args.rel, "r", "utf-8") as rel:
        for line in rel:
            line = line.strip('\n').split()
            id2rel[line[1]] = line[0]

    with codecs.open(args.id_file, "r", "utf-8") as res, codecs.open(args.test, "w", "utf-8") as out:
        index = 1
        for line in res:
            line = line.strip('\n').split('\t')
            out.write(str(8000 + index) + '\t' + id2rel[line[0]] + '\n')
            index += 1
