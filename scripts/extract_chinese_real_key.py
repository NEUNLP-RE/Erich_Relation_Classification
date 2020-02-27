from __future__ import print_function

import argparse
import codecs


def add_parameters(description):
    params = argparse.ArgumentParser(description=description)
    params.add_argument("--source_file", "-s", required=True, help="the file need to split")
    params.add_argument("--output_file", "-o", required=True, help="the output file")
    return params


if __name__ == "__main__":
    parsers = add_parameters("extract real chinese key file")
    args = parsers.parse_args()

    with codecs.open(args.source_file, "r", "utf-8") as fs, codecs.open(args.output_file, "w", "utf-8") as fo:
        for l in fs:
            blocks = l.strip().split()
            rels = blocks[0].split(",")
            fo.write(" ".join(rels) + "\n")

    print("End...")
