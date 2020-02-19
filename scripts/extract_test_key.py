from __future__ import print_function

import argparse
import codecs


def add_parameters(description):
    params = argparse.ArgumentParser(description=description)
    params.add_argument("--sem_test_file", "-s", required=True, help="source semeval dataset test file")
    params.add_argument("--key_file", "-k", required=True, help="output test key file")
    return params


if __name__ == "__main__":
    parsers = add_parameters("extract test key from test file in semeval")
    args = parsers.parse_args()

    with codecs.open(args.sem_test_file, "r", "utf-8") as fs:
        text = fs.readlines()
        test_key = [t.strip() for t in text[1::4]]

    with codecs.open(args.key_file, "w", "utf-8") as fk:
        for i in range(len(test_key)):
            fk.write(str(8000 + i + 1) + "\t" + test_key[i] + "\n")
