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
from copy import deepcopy

from dreamcoder.utilities import *
from dreamcoder.domains.regex.groundtruthRegexes import *
from dreamcoder.program import Abstraction, Application

from dreamcoder.domains.regex.makeRegexTasks import regexHeldOutExamples
from dreamcoder.domains.regex.regexPrimitives import PRC

# import torch
# from torch.nn import Parameter
# from torch import optim

# import torch.nn.functional as F
import string

from pathlib import Path




from dreamcoder.program import Program
from dreamcoder.domains.list.listPrimitives import bootstrapTarget_extra
bootstrapTarget_extra() # we need to do this bc it has side effects that make parsing list programs work lol


def parse_grammar(text, domain):
    assert text.startswith('Grammar after iteration')
    text = text[text.index('\n')+1:] # skip to the next line

    # print(f'parsing from {text[:3000]}')

    lines = text.split('\n')

    continuationType = None

    if lines[0].startswith('continuation'):
        lines = lines[1:]
        assert domain in ('towers','logo')
        if domain == 'towers':
            continuationType = {
                "constructor": "tower",
                "arguments": []
            }
        if domain == 'logo':
            continuationType = {
                "constructor": "turtle",
                "arguments": []
            }


    logVariable,type,prim = lines[0].split('\t')
    assert prim == '$_'
    assert type == 't0'

    productions = []

    for line in lines[1:]:
        # try:
        res = line.split('\t')
        if len(res) not in (3,4):
            break
        if len(res) == 3:
            score,type,prim = res
        if len(res) == 4:
            score,type,prim,_eval = res
        productions.append({'expression':prim, 'logProbability':float(score)})
        # except ValueError:
        #     break # we break when it fails to parse into a 3 item line... (bc there's weird variation in the log file format)

    assert len(productions) > 3 # just some reasonable number bc try/except is scary!

    # print(f'parsed grammar: {len(productions)}')

    grammar = {
        'logVariable': float(logVariable),
        'productions': productions
    }

    if continuationType is not None:
        grammar['continuationType'] = continuationType

    return grammar


def concretize_request(request,domain):
    # turn any t0 into a domain specific default lmao

    if 'arguments' in request:
        for arg in request['arguments']:
            concretize_request(arg,domain)
    
    if request == {"index": 0}:
        del request["index"]
        if domain == 'text':
            request["constructor"] = "char"
            request["arguments"] = []
        elif domain in ('list','origami'):
            request["constructor"] = "int"
            request["arguments"] = []
        elif domain == 'rational':
            request["constructor"] = "real"
            request["arguments"] = []
        else:
            assert False, f"domain not handled {domain}"
            


    



if __name__ == "__main__":

    CPUS = 8 # todo hardcoded   
    
    log_file = Path(str(sys.argv[1]))
    domain = str(sys.argv[2])
    run = str(sys.argv[3])
    out_dir = Path(str(sys.argv[4]))

    with open(log_file, 'r') as f:
        log = f.read()
    
    assert 'Running EC on' in log
    assert 'cuda  =' in log
    assert 'Grammar after iteration' in log
    assert 'Showing the top 5 programs in each frontier' in log
    assert 'Exported primitive graph' in log
    assert 'Improved score to' in log

    parameters = log[log.index('Running EC on'): log.index('cuda  =')]
    parameters = {p.split('=')[0].strip(): p.split('=')[1].strip() for p in parameters.split('\n') if '=' in p.strip()}
    assert 'arity' in parameters
    assert 'aic' in parameters
    assert 'pseudoCounts' in parameters
    assert 'structurePenalty' in parameters
    assert 'topK' in parameters
    assert 'iterations' in parameters
    assert '\n\nCompression message saved to' in log

    topK = int(parameters['topK'])

    # template of message without grammar + frontiers filled in
    message_template = {
        # 'fast_final_rewrite': True, # flag for https://github.com/mlb2251/stitch_dreamcoder 
        "verbose": False,
        "arity": int(parameters['arity']),
        "topK": topK,
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

    bench_num = 0

    # initial grammar is just some later grammar but then throw out the learned inventions (which start with "#")
    initial_grammar = parse_grammar(log[log.index('Grammar after iteration'):], domain)
    initial_grammar['logVariable'] = 0.0
    initial_grammar['productions'] = [{'expression':prod['expression'], 'logProbability':0.0} for prod in initial_grammar['productions'] if not prod['expression'].startswith('#')]

    # print('Initial grammar:', initial_grammar)
    # len_grammar = len(initial_grammar['productions'])

    iterations = log.split('Showing the top 5 programs in each frontier')[1:]

    for i,iteration in enumerate(iterations):
        # print(f"i: {i} input grammar len: {len(initial_grammar['productions'])}")
        # get the programs being sent
        iteration = iteration[iteration.index('\n')+1:] # skip to the next line
        programs_chunk = iteration[:iteration.index('\n\nCompression message saved to')]

        frontiers = []
        for task in programs_chunk.split('\n\n'):
            task = task.split('\n')
            task_name = task[0]
            assert len(task) <= 6 # these are top 5 programs after all
            programs = []
            for prog in task[1:]:
                score = float(prog[:prog.index('(')].strip())
                program = prog[prog.index('('):].strip()
                # todo big warning: this is not the logLikelihood this is just 0
                programs.append({'program': program, 'logLikelihood': 0.0 })

            programs = programs[:topK] # only keep the topK programs
            request = Program.parse(programs[0]['program']).infer() # grab the request off of some program
            request_json = request.json()
            assert 't1' not in str(request)
            if 't0' in str(request):
                concretize_request(request_json,domain)
            frontiers.append(
                {"request": request_json,
                 "programs": programs})
        
        new_grammar = parse_grammar(iteration[iteration.index(f'Grammar after iteration {i+1}'):], domain)

        num_learned = len(new_grammar['productions']) - len(initial_grammar['productions'])
        assert num_learned >= 0
        if num_learned > 0:
            # we learned something hooray!
            print(f'Learned {num_learned} things!')
            
            message = deepcopy(message_template)
            message['DSL'] = initial_grammar
            message['frontiers'] = frontiers
            message['info'] = {
                'iteration': i,
                'num_learned': num_learned,
                'new_grammar': new_grammar,
            }

            out_file = out_dir / f'bench{bench_num:03d}_it{i}.json'

            with open(out_file, 'w') as f:
                json.dump(message, f, indent=4)
            print(f'Wrote {out_file}')
            bench_num += 1

        assert len(new_grammar['productions']) >= len(initial_grammar['productions'])
        
        # the initial grammar for the next iteration is our final grammar
        initial_grammar = new_grammar




    
    print("done")



