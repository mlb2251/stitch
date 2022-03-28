###############################################################################
# This file is a utility for extracting compression messages from the
# DreamCoder PLDI artifact.
# Before running it, place it in the bin/ directory of the artifact.
# TODO(theoxo): Some of the imports below may not be necessary to load the
# pickle; I haven't bothered finding out which.
###############################################################################
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

import re
import json
import pregex as pre
import sys
import os

from dreamcoder.utilities import *
from dreamcoder.domains.regex.groundtruthRegexes import *
from dreamcoder.program import Abstraction, Application

from dreamcoder.domains.regex.makeRegexTasks import regexHeldOutExamples
from dreamcoder.domains.regex.regexPrimitives import PRC

import torch
from torch.nn import Parameter
from torch import optim

import torch.nn.functional as F
import string

if __name__ == "__main__":
    
    checkpoint_file = str(sys.argv[1])
    with open(checkpoint_file, 'rb') as file:
        checkpoint = pickle.load(file)


    # This is the fn Kevin used in his email. The .frontiersOverTime field used here isn't actually
    # defined in all of the log files.
    def frontiers_at_iteration(i): return {task: checkpoint.frontiersOverTime[task][i] for task in checkpoint.frontiersOverTime.keys()}

    messages = []  # one compression message per iteration
    for idx, g in enumerate(checkpoint.grammars[:-1]):
        # Note above that I am not saving the last grammar. This is because, as I understand things,
        # the very last grammar is the grammar after _all_ of compression, and so does not have any
        # frontiers associated with it (and is thus irrelevant for our teacher-forcing purposes?)
        assert (all([len(checkpoint.grammars) - 1 == len(checkpoint.frontiersOverTime[t]) for t in checkpoint.frontiersOverTime.keys()]))
        message = {
                "verbose": False,
                "arity": checkpoint.parameters['arity'],
                "topK": checkpoint.parameters['topK'],
                "pseudoCounts": float(re.search(r'pc=([^_]+)', checkpoint_file).group(1)),
                "aic": float(re.search(r'aic=([^_]+)', checkpoint_file).group(1)),
                "bs": 1000000,  # the Ocaml backend always uses these values for bs and topI I think;
                "topI": 300,    # see lines 53-54 in compression.py
                "structurePenalty": float(re.search(r'_L=([^_]+)', checkpoint_file).group(1)),
                "CPUs": int(sys.argv[3]),     
                "lc_score": 0.0,     # weight required by LAPS, but irrelevant for our purposes
                "DSL": g.json(),
                "iterations": checkpoint.parameters['iterations'],
                "frontiers": [f.json() for f in frontiers_at_iteration(idx).values() if len(f.json()['programs']) > 0],
        }
        messages.append(json.dumps(message, indent=4))

    out_dir_name = sys.argv[2]
    os.makedirs(out_dir_name, exist_ok=True)
    for idx, msg in enumerate(messages):
        with open(f'{out_dir_name}/iteration_{idx}.json', 'w') as outf:
            outf.write(msg)
        print(f'Wrote json msg for iteration {idx}')
