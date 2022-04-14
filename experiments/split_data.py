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
    test_split = 250 - int(20*2.5)  # Test set size is fixed to 20% of programs
    train_split = int(split * float(test_split)/100.)

    assert train_split <= test_split  # no overlap of train/test allowed
    train_progs = programs[:train_split]
    test_progs = programs[:test_split]

    with open(outf, 'w') as out:
        json.dump([train_progs, test_progs], out)

