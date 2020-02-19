from __future__ import print_function

import argparse
import codecs
import re
from stanfordcorenlp import StanfordCoreNLP


def add_parameters(descriptions):
    params = argparse.ArgumentParser(description=descriptions)
    params.add_argument("--sem_file", "-s", required=True, help="source file of Semeval dataset")
    params.add_argument("--token_dir", "-t", required=True, help="stanford tokenizer dir")
    params.add_argument("--rel", "-r", required=True, help="relation2id")
    params.add_argument("--format_file", "-f", required=True, help="format file for model input")
    return params


def parse_sem_data(orig_txt, nlp_toolkit):
    new_txt = " ".join(nlp_toolkit.word_tokenize(orig_txt))
    # "<ex> word </ex>" -> "<ex>word</ex>"
    new_txt = re.sub(r"<e([12])> (.*?) </e\1>", r"<e\1>\2</e\1>", new_txt)
    # get pos by "char"
    e1_char_pos = list(re.search("<e1>.*?</e1>", new_txt).span())
    e2_char_pos = list(re.search("<e2>.*?</e2>", new_txt).span())
    e1_p = [len(new_txt[:e1_char_pos[0]].strip().split()), len(new_txt[:e1_char_pos[1]].strip().split()) - 1]
    e2_p = [len(new_txt[:e2_char_pos[0]].strip().split()), len(new_txt[:e2_char_pos[1]].strip().split()) - 1]
    res_txt = re.sub(r"<e([12])>(.*?)</e\1>", r"\2", new_txt).split()
    return e1_p, e2_p, res_txt


if __name__ == "__main__":
    parsers = add_parameters("convert source file in semeval dataset to format file")
    args = parsers.parse_args()

    with codecs.open(args.sem_file, "r", "utf=8") as fs:
        lines = [l.strip() for l in fs.readlines()]

    rel2ids = {}
    with codecs.open(args.rel, "r", "utf-8") as fr:
        for l in fr:
            rel, ids = l.strip().split("\t")
            rel2ids[rel] = ids

    assert len(lines) % 4 == 0
    # load stanford nlp toolkit
    stanford_nlp = StanfordCoreNLP(args.token_dir)
    dataset = []
    for i in range(0, len(lines), 4):
        txt_id, text = lines[i].split("\t")
        # assert int(txt_id) == i/4 + 1
        e1_pos, e2_pos, txt = parse_sem_data(text[1:-1], stanford_nlp)
        rel_id = rel2ids[lines[i + 1].strip()]
        str_line = " ".join([str(rel_id)] + ["%s" % p for p in e1_pos] +
                            ["%s" % p for p in e2_pos] + txt)
        dataset.append(str_line)
        print("\rProcess %d" % (i/4 + 1), end="")
    with codecs.open(args.format_file, "w", "utf-8") as ff:
        for d in dataset:
            ff.write(d + "\n")
    stanford_nlp.close()
    print("\nEnd...")
