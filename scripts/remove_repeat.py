from __future__ import print_function

import sys

with open(sys.argv[1], "r", encoding="utf-8") as f1, open(sys.argv[2], "w", encoding="utf-8") as f2:
    sents = []
    index = 0
    for l in f1:
        if l.strip() not in sents:
            sents.append(l.strip())
            f2.write(l)
        print("\rProcess %d" % index, end="")
        index += 1
