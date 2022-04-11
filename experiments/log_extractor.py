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

from pathlib import Path

if __name__ == "__main__":

    CPUS = 8 # todo hardcoded   
    
    log_file = Path(str(sys.argv[1]))
    domain = str(sys.argv[2])
    run = str(sys.argv[3])
    out_file = Path(str(sys.argv[4]))
    assert str(out_file).endswith('.json')

    with open(log_file, 'r') as f:
        log = f.read()
    
    assert 'Running EC on' in log
    assert 'cuda  =' in log
    assert 'Grammar' in log
    assert 'Showing the top 5 programs in each frontier' in log
    assert 'Exported primitive graph' in log

    parameters = log[log.index('Running EC on'): log.index('cuda  =')]
    parameters = {p.split('=')[0].strip(): p.split('=')[1].strip() for p in parameters.split('\n') if '=' in p.strip()}
    assert 'arity' in parameters
    assert 'aic' in parameters
    assert 'pseudoCounts' in parameters
    assert 'structurePenalty' in parameters
    assert 'topK' in parameters
    assert 'iterations' in parameters

    # template of message without grammar + frontiers filled in
    message_template = {
        "verbose": False,
        "arity": int(parameters['arity']),
        "topK": int(parameters['topK']),
        "pseudoCounts": float(parameters['pseudoCounts']),
        "aic": float(parameters['aic']),
        "bs": 1000000,  # the Ocaml backend always uses these values for bs and topI I think;
        "topI": 300,    # see lines 53-54 in compression.py
        "structurePenalty": float(parameters['structurePenalty']),
        "CPUs": CPUS,
        "lc_score": 0.0,     # weight required by LAPS, but irrelevant for our purposes
        "DSL": None,
        "iterations": int(parameters['iterations']),
        "frontiers": None,
    }
    
    print("done")

    exit(0)
    


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
