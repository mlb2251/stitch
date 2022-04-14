import json
import sys
import random

split = int(sys.argv[1])
seed  = int(sys.argv[2])
dataf = sys.argv[3]
outf  = sys.argv[4]

random.seed(seed)

with open(dataf, 'rb') as data:
    programs = json.load(data)
    assert len(programs) == 250

    random.shuffle(programs)

    # split is % of programs that should be put
    # into the training set.
    # Since we have 250 programs per input file,
    # we need to multiply this by 2.5 to get
    # the number of programs to put in the
    # training set.
    split_index = int(split * 2.5)
    train_progs = programs[:split_index]
    test_progs = programs[split_index:]

    with open(outf, 'w') as out:
        json.dump([train_progs, test_progs], out)

