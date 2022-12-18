import json
import sys
import random

seed  = int(sys.argv[1])
dataf = sys.argv[2]
outf  = sys.argv[3]

random.seed(seed)

with open(dataf, 'rb') as data:
    programs = json.load(data)
    assert len(programs) == 250

    random.shuffle(programs)

    split = 200  # Test set size is fixed to 20% of programs

    train_progs = programs[:split]
    test_progs = programs[split:]

    with open(outf, 'w') as out:
        json.dump([train_progs, test_progs], out)

