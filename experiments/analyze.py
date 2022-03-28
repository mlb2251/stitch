import json
import sys
import os
import re
from typing import *
from pathlib import Path

"""
This converts from any dreamcoder json that can be indexed like json["DSL"]["productions"]["expression]
to get a program into various stitch formats. For example this works for
compression messages, ./compression stdout outputs, and ./compression inputs.

The output is a json that has a list with the following keys (also see example below):
    name: the name of the primitive eg fn_0
    dreamcoder: the dreamcoder style string (like the "dreamcoder" field of stitch format)
    with_sub_inventions: the dreamcoder string but with inner inventions replaced with fn_i calls instead of inlining their bodies
    stitch_uncanonical: stitch format using ivars and lams. Note that this is not canonical meaning #1 might come before #0
    stitch_canonical: canonicalized so #0 comes before #1 etc.
        Useful for seeing if stitch can find the same invention
        *HOWEVER* reordering the ivars means any rewritten frontiers with this will be nonsensical, so
        use stitch_uncanonical if you need to actually maintain semantics of frontiers, do multiuse scoring, etc
    arity: arity (since stitch needs to know this)

Example:
    name: fn_0
    dreamcoder: #(lambda (lambda (lambda (logo_forLoop $0 (lambda (lambda (logo_FWRT $4 $3 $0)))))))
    with_sub_inventions: #(lambda (lambda (lambda (logo_forLoop $0 (lambda (lambda (logo_FWRT $4 $3 $0)))))))
    stitch_uncanonical: (logo_forLoop #2 (lam (lam (logo_FWRT #0 #1 $0))))
    stitch_canonical: (logo_forLoop #0 (lam (lam (logo_FWRT #1 #2 $0))))
    arity: 3

Another example:
    name: fn_5
    dreamcoder: #(lambda (logo_forLoop 7 (lambda (lambda (#(lambda (lambda (lambda (logo_forLoop $0 (lambda (lambda (logo_FWRT $4 $3 $0))))))) (logo_MULL logo_epsL $2) logo_epsA 7 $0)))))
    with_sub_inventions: #(lambda (logo_forLoop 7 (lambda (lambda (fn_0 (logo_MULL logo_epsL $2) logo_epsA 7 $0)))))
    stitch_uncanonical: (logo_forLoop 7 (lam (lam (fn_0 (logo_MULL logo_epsL #0) logo_epsA 7 $0))))
    stitch_canonical: (logo_forLoop 7 (lam (lam (fn_0 (logo_MULL logo_epsL #0) logo_epsA 7 $0))))
    arity: 1

"""

RUNS = {
    'list': [
        '2019-02-15T16:31:58.353555',
        '2019-02-15T16:36:06.861462',
        '2019-02-15T16:39:37.056521',
        '2019-02-15T16:43:47.195104',
        '2019-02-15T16:47:54.339779',
    ],
    'logo': [
        '2019-03-23T18:06:23.106382',
        '2019-03-23T18:10:01.834307',
        '2019-03-23T18:13:43.818837',
        '2019-03-23T18:17:35.090237',
        '2019-03-23T18:21:31.435612',
        '2019-11-29T16:33:12.597455',
    ],
    'rational': [
        '2019-02-15T11:28:17.171165',
        '2019-02-15T11:28:20.454707',
        '2019-02-15T11:28:27.615614',
        '2019-02-15T11:28:40.635813',
        '2019-02-19T18:04:34.743890',
        '2019-11-29T16:56:44.307505',
    ],
    'rec-fp': [
        'iteration=10_2019-07-11T19:49:10.899159',
        'iteration=1_2019-07-11T19:49:10.899159',
        'iteration=2_2019-07-11T19:49:10.899159',
        'iteration=3_2019-07-11T19:49:10.899159',
        'iteration=4_2019-07-11T19:49:10.899159',
        'iteration=5_2019-07-11T19:49:10.899159',
        'iteration=6_2019-07-11T19:49:10.899159',
        'iteration=7_2019-07-11T19:49:10.899159',
        'iteration=8_2019-07-11T19:49:10.899159',
        'iteration=9_2019-07-11T19:49:10.899159',
    ],
    'regex': [
        '2019-03-04T19:13:10.698186',
        '2019-03-04T19:17:09.170192',
        '2019-03-04T19:21:20.153727',
        '2019-03-04T19:25:56.137631',
        '2019-03-04T19:30:01.252339',
    ],
    'text': [
        '2019-01-25T02:53:59.182941',
        '2019-01-25T02:58:17.082195',
        '2019-01-25T03:02:10.385246',
        '2019-01-25T03:06:06.834290',
        '2019-01-25T03:09:45.988462',
        '2019-01-26T01:23:35.287813',
    ],
}



def load(file):
    with open(file,'rb') as f:
        return json.load(f)

def save(obj, file):
    with open(file,'w') as f:
        json.dump(obj,f,indent=4)
    print(f"wrote {file}")

def stitch_format(with_sub_inventions):
    source = with_sub_inventions[1:] # remove '#'
    assert source.count('#') == 0

    arity = 0
    while source.startswith('(lambda '):
        source = source[len('(lambda '):]
        source = source[:-1] # remove ')'
        arity += 1

    # stack = [i for i in range(arity)]
    stack = []
    res = ''
    # source looks like: "(lambda (lambda (fn_0 $1 $0 logo_IFTY)))"
    while source != '':
        # print(source)
        if source.startswith('(lambda '):
            source = source[len('(lambda '):]
            stack.append('lambda')
            res += '(lambda '
        elif source.startswith('('):
            source = source[1:]
            stack.append('paren')
            res += '('
        elif source.startswith(')'):
            source = source[1:]
            stack.pop() # remove a lambda or a paren
            res += ')'
        elif source.startswith('$'):
            source = source[1:]
            var = ''
            while source != '' and source[0].isdigit():
                var += source[0]
                source = source[1:]
            var = int(var)
            lamdba_depth = len([s for s in stack if s == 'lambda'])
            upward_ref = var - lamdba_depth
            if upward_ref < 0:
                # internal reference not to an ivar
                res += f'${var}'
            else:
                # ivar!
                ivar = arity - upward_ref - 1
                res += '#' + str(ivar)
                assert ivar < arity
        else:
            res += source[0]
            source = source[1:]

    res = res.replace('lambda','lam')
    return res,arity

def canonicalize(uncanonical):
    res = uncanonical
    i = 100
    while i >= 0:
        res = res.replace('#'+str(i), f'#_#_{i}#_#_')
        i -= 1 # go in decreasing order so #100 doesnt get rewritten by #1 etc

    # get ordering of first instance of each ivar
    ivars = []
    for ivar in res.split('#_#_'):
        if ivar.isdigit() and ivar not in ivars:
            ivars.append(int(ivar))

    i = 100
    while i >= 0:
        if i in ivars:
            # for example if #2 was the first ivar to appear then itd get rewritten to #0
            res = res.replace(f'#_#_{i}#_#_', '#' + str(ivars.index(i)))
        i -= 1 # go in decreasing order so #100 doesnt get rewritten by #1 etc
    
    return res


# pull out all the learned library fns
def to_stitch_dsl(dc_json):
    dsl_dc = [prod["expression"] for prod in dc_json["DSL"]["productions"]]
    dsl_dc = [p for p in dsl_dc if p.startswith("#")]
    dsl_dc.reverse() # reverse so first one learned comes first
    dsl = []
    for i,dc_string in enumerate(dsl_dc):
        with_sub_inventions = dc_string
        for entry in dsl[::-1]: # reverse so we rewrite with larger inventions first to not mangle the internals
            with_sub_inventions = with_sub_inventions.replace(entry['dreamcoder'], entry['name'])
        assert with_sub_inventions.count("#") == 1

        stitch_uncanonical,arity = stitch_format(with_sub_inventions)
        stitch_canonical = canonicalize(stitch_uncanonical)

        dsl.append({'name':f'fn_{i}','dreamcoder':dc_string, 'with_sub_inventions':with_sub_inventions, 'stitch_uncanonical':stitch_uncanonical, 'stitch_canonical':stitch_canonical, 'arity':arity})
    return dsl

def to_stitch_program(program: str, stitch_dsl):
    for entry in stitch_dsl[::-1]:
        program = program.replace(entry['dreamcoder'], entry['name'])
    program = program.replace('lambda','lam')
    assert '#' not in program
    return program

COST_NONTERMINAL = 1
COST_TERMINAL = 100
def stitch_cost(stitch_program):
    cost = 0
    # lambda costs
    cost += COST_NONTERMINAL * stitch_program.count('(lam ')
    stitch_program = stitch_program.replace('(lam ','')
    # app costs are based on spaces now that we've removed the lam space
    cost += COST_NONTERMINAL * stitch_program.count(' ')
    # clear parens 
    stitch_program = stitch_program.replace('(','')
    stitch_program = stitch_program.replace(')','')
    # prim/var costs is the number of space separated things remaining
    cost += COST_TERMINAL * len([x for x in stitch_program.split(' ') if x != ''])
    return cost

def dreamcoder_to_invention_info(in_file, out_file):
    in_json = load(in_file)
    out_json = load(out_file)
    stitch_dsl_input = to_stitch_dsl(in_json)
    stitch_dsl_output = to_stitch_dsl(out_json)
    new_fn = diff(stitch_dsl_input,stitch_dsl_output)
    assert len(new_fn) == 1, f"these inputs and outputs differ by {len(new_fn)} functions (must differ by 1 fn)"
    new_fn = new_fn[0]
    
    all_programs_out = [programs['program'] for f in out_json['frontiers'] for programs in f['programs']]
    stitch_programs_out = [to_stitch_program(p,stitch_dsl_output) for p in all_programs_out]
    stitch_programs_cost_out = sum([stitch_cost(p) for p in stitch_programs_out])
    
    all_programs_in = [programs['program'] for f in in_json['frontiers'] for programs in f['programs']]
    stitch_programs_in = [to_stitch_program(p,stitch_dsl_input) for p in all_programs_in]
    stitch_programs_cost_in = sum([stitch_cost(p) for p in stitch_programs_in])
    
    compressive_utility = (stitch_programs_cost_in - stitch_programs_cost_out)
    compressive_multiplier = stitch_programs_cost_in / stitch_programs_cost_out

    return {
        'in_file': str(in_file),
        'out_file': str(out_file),
        'name': new_fn['name'],
        'stitch_uncanonical': new_fn['stitch_uncanonical'],
        'stitch_canonical': new_fn['stitch_canonical'],
        'dreamcoder': new_fn['dreamcoder'],
        'dreamcoder_frontiers_score': None,
        'stitch_programs_cost': stitch_programs_cost_out,
        'compressive_utility': compressive_utility,
        'compressive_multiplier': compressive_multiplier,
        'stitch_utility': None,
        'usages': usages(new_fn["name"], stitch_programs_out),
        'stitch_programs': stitch_programs_out,
        'dreamcoder_frontiers': out_json['frontiers'],
    }

def stitch_to_invention_info(stitch_json):
    in_json = load(stitch_json['args']['file'])
    stitch_dsl_input = to_stitch_dsl(in_json)

    assert len(stitch_json['invs']) == 1, "there seem to be more than one invention in this file"
    inv = stitch_json['invs'][0]

    all_programs_in = [programs['program'] for f in in_json['frontiers'] for programs in f['programs']]
    stitch_programs_in = [to_stitch_program(p,stitch_dsl_input) for p in all_programs_in]
    stitch_programs_cost_in = sum([stitch_cost(p) for p in stitch_programs_in])
    compressive_utility = (stitch_programs_cost_in - inv['final_cost'])

    assert stitch_programs_cost_in == stitch_json["original_cost"]
    assert compressive_utility ==  stitch_json["original_cost"] - inv["final_cost"]

    return {
            'in_file': stitch_json['args']['file'],
            'out_file': stitch_json['args']['out'],
            'name': inv['name'],
            'stitch_uncanonical': inv['body'],
            'stitch_canonical': inv['body'],
            'dreamcoder': inv['dreamcoder'],
            'dreamcoder_frontiers_score': None,
            'stitch_programs_cost': inv['final_cost'],
            'compressive_utility': stitch_json["original_cost"] - inv["final_cost"],
            'compressive_multiplier': inv["multiplier"],
            'stitch_utility': inv["utility"],
            'usages': inv['num_uses'],
            'stitch_programs': inv['rewritten'],
            'dreamcoder_frontiers': None,
        }


def usages(fn_name, stitch_programs):
    # we count name + closeparen or name + space so that fn_1 doesnt get counted for things like fn_10
    return sum([p.count(f'{fn_name})') + p.count(f'{fn_name} ') for p in stitch_programs])


def diff(input_dsl,output_dsl):
    # print("\n\nIN\n\n")
    # for k in input_dsl:
    #     print(f'{k["name"]}: {k["dreamcoder"]}')
    # print("\n\nOUT\n\n")
    # for k in output_dsl:
    #     print(f'{k["name"]}: {k["dreamcoder"]}')
    difference = len(output_dsl) - len(input_dsl)
    assert difference >= 0
    if difference != 0:
        assert input_dsl == output_dsl[:-difference]
        return output_dsl[-difference:]
    else:
        assert input_dsl == output_dsl
        return []

def run_info(dir):
    runs = []
    for file in os.listdir(dir):
        m = re.match(r'iteration_(\d*)\.json',file)
        if m is None:
            print("skipping",file)
            continue
        iteration = int(m.group(1))
        file = os.path.join(dir,file)
        dc_json = load(file)
        dsl = to_stitch_dsl(dc_json)
        runs.append({
            'iteration':iteration,
            'file':file,
            'dsl_len': len(dsl),
            'num_tasks_solved': len(dc_json["frontiers"]),
            'num_programs': len([p for f in dc_json["frontiers"] for p in f['programs']]),
            })
    runs.sort(key=lambda x: x['iteration'])
    assert [x['iteration'] for x in runs] == list(range(len(runs)))
    for i in range(len(runs)-1):
        dsl = to_stitch_dsl(load(runs[i]['file']))
        dsl_next = to_stitch_dsl(load(runs[i+1]['file']))
        dsl_diff = diff(dsl,dsl_next)
        runs[i]['num_expected_new_fns'] = len(dsl_diff)
        runs[i]['expected_new_fns'] = dsl_diff
    runs = runs[:-1] # ignore the last one that we couldn't diff on

    num_expected_new_fns = [ x['num_expected_new_fns'] for x in runs ]
    num_tasks_solved = [ x['num_tasks_solved'] for x in runs ]
    num_programs = [ x['num_programs'] for x in runs ]
    res = {
        'num_expected_new_fns': num_expected_new_fns,
        'total_new_fns': sum(num_expected_new_fns),
        'num_tasks_solved': num_tasks_solved,
        'num_programs': num_programs,
        'iterations':runs,
    }
    return res

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'dsl':
        save(to_stitch_dsl(load(sys.argv[2])), sys.argv[3])

    elif mode == 'diff':
        input_dsl = to_stitch_dsl(load(sys.argv[2]))
        output_dsl = to_stitch_dsl(load(sys.argv[3]))
        difference = diff(input_dsl,output_dsl)
        print(f"{len(difference)} new dsl functions in diff")
        save(difference, sys.argv[4])

    elif mode == 'run_info':
        save(run_info(sys.argv[2]), os.path.join(sys.argv[2],'info.json'))
        print("wrote", os.path.join(dir,'info.json'))

    elif mode == 'all_run_info':
        for domain,runs in RUNS.items():
            for run in runs:
                save(run_info(os.path.join('data',domain,run)), os.path.join('data',domain,run,'info.json'))
    
    elif mode == 'invention_info_dc':
        in_file = sys.argv[2]
        out_file = sys.argv[3]
        out_path = Path(sys.argv[3]).with_suffix('.invention_info.json')
        save(dreamcoder_to_invention_info(in_file,out_file), out_path)

    elif mode == 'run_invention_info_dc':
        data_path = Path(sys.argv[2])
        out_path = Path(sys.argv[3])
        save_dir = out_path / 'invention_info'
        save_dir.mkdir(exist_ok=True)
        input_run_info = load(data_path / 'info.json')

        summary_json = []

        for i in range(0,len(input_run_info['iterations'])):
            inv = 0
            curr_input = data_path / f'iteration_{i}.json'
            for file in sorted(os.listdir(out_path / f'iteration_{i}_rerun_compressionMessages')):
                output = out_path / f'iteration_{i}_rerun_compressionMessages' / file
                inv_info = dreamcoder_to_invention_info(curr_input, output)
                save(inv_info, save_dir / f'iteration_{i}_inv{inv}.json')
                inv_info['dreamcoder_frontiers'] = None
                inv_info['stitch_programs'] = None
                summary_json.append(inv_info)
                curr_input = output
                inv += 1
        save(summary_json, save_dir / f'info.json')
    
    elif mode == 'to_input_files':
        out_path = Path(sys.argv[2])
        in_files = [x['in_file'] for x in load(out_path / 'invention_info' / 'info.json')]
        for i,in_file in enumerate(in_files):
            print(in_file)
    
    elif mode == 'run_invention_info_stitch':
        out_path = Path(sys.argv[2])
        save_dir = out_path / 'stitch' / 'invention_info'
        save_dir.mkdir(exist_ok=True)
        summary = []
        for file in sorted([f for f in os.listdir(out_path / 'stitch') if f.endswith('.json')], key=lambda x: int(re.match(r'out_(\d*)',x).group(1))):
            stitch_json = load(out_path / 'stitch' / file)
            inv_info = stitch_to_invention_info(stitch_json)
            save(inv_info, save_dir / f'{file}.json')
            inv_info['dreamcoder_frontiers'] = None
            inv_info['stitch_programs'] = None
            summary.append(inv_info)
        save(summary, save_dir / 'info.json')

    elif mode == 'compare':
        out_path = Path(sys.argv[2])
        stitch = load(out_path / 'stitch' / 'invention_info' / 'info.json')
        dc = load(out_path / 'invention_info' / 'info.json')
        for i,(s,d) in enumerate(zip(stitch,dc)):
            assert s['name'] == d['name']
            assert s['in_file'] == d['in_file']
            if s['compressive_utility'] == d['compressive_utility']:
                print(f"{i}: exact match for compressive_utility")
            elif s['compressive_utility'] < d['compressive_utility']:
                print(f"{i}: WARNING STITCH IS WORSE IN compressive_utility")
                print("===STITCH===")
                for k,v in s.items():
                    print(f"{k}: {v}")
                print("===DREAMCODER===")
                for k,v in d.items():
                    print(f"{k}: {v}")



"""
Unified single step output format for exactly 1 new invention
invention_info.json

name
stitch_uncanonical
stitch_canonical
dreamcoder
dreamcoder_frontiers_score
stitch_frontiers_cost
compressivity
stitch_utility (null for now)
stitch_frontiers
dreamcoder_frontiers

"""



    

