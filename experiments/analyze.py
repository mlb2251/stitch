import json
import sys
import os
import re
from typing import *
from pathlib import Path
import shutil
from copy import deepcopy
from stitch_core import *

def load(file):
    with open(file,'rb') as f:
        return json.load(f)

def save(obj, file):
    with open(file,'w') as f:
        json.dump(obj,f,indent=4)
    print(f"wrote {file}")

def stitch_format(with_sub_inventions):
    """
    takes an invention in dreamcoder format that does NOT have nested inventions (ie they've already
    been rewritten from `#(...)` to `fn_i`) and rewrites it into stitch format. Specifically:
    - remove any `(lambda (lambda (...)))` wrapping the whole invention, treating that as the arity
    - lambda -> lam
    - $i -> #j if it's a reference to one of the lambdas that used to wrap the invention
    """
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
    """
    takes a stitch invention and permutes the #i indices
    such that forall j and k, if j<k then the first instance of #j always comes before the first instance of #k
    """
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

def extract_inventions_dreamcoder(dc_json):
    """
    extract the learned inventions from a dreamcoder json file and sort by length
    """
    return sorted([prod["expression"] for prod in dc_json["DSL"]["productions"] if prod["expression"].startswith("#")], key=len)

def stitch_json_to_stitch_invs(stitch_json):
    return [{
        'name': inv['name'],
        'dreamcoder': inv['dreamcoder'],
        'stitch_uncanonical': inv['body'],
        'stitch_canonical': canonicalize(inv['body']),
        'arity': inv['arity'],
    } for inv in stitch_json['abstractions']]


# pull out all the learned library fns
def to_stitch_invs(dc_invs, start_from=None):
    """
    Converts from a list of '#(...)' dreamcoder inventions to a list of stitch inventions. Optionally primed with a list of stitch inventions thru `start_from`.

    Assumes start_from is in order of increasing complexity so later inventions build on earlier ones

    * Important: since dreamcoder doesnt always preserve the order of learned functions in its dsl output object, we destroy that ordering by sorting here. However we dont destroy the order of start_from *

    The output is a json that has a list with the following keys (also see example below):
        name: the name of the primitive eg fn_0
        dreamcoder: the dreamcoder style string (like the "dreamcoder" field of stitch format)
        stitch_uncanonical: stitch format using ivars and lams. Note that this is not canonical meaning #1 might come before #0
        stitch_canonical: canonicalized so #0 comes before #1 etc.
            Useful for seeing if stitch can find the same invention
            *HOWEVER* reordering the ivars means any rewritten frontiers with this will be nonsensical since the argument orders will have changed, so
            use stitch_uncanonical if you need to actually maintain semantics of frontiers, do multiuse scoring, etc
        arity: arity (since stitch needs to know this)

    Example 1:
        name: fn_0
        dreamcoder: #(lambda (lambda (lambda (logo_forLoop $0 (lambda (lambda (logo_FWRT $4 $3 $0)))))))
        stitch_uncanonical: (logo_forLoop #2 (lam (lam (logo_FWRT #0 #1 $0))))
        stitch_canonical: (logo_forLoop #0 (lam (lam (logo_FWRT #1 #2 $0))))
        arity: 3

    Example 2:
        name: fn_5
        dreamcoder: #(lambda (logo_forLoop 7 (lambda (lambda (#(lambda (lambda (lambda (logo_forLoop $0 (lambda (lambda (logo_FWRT $4 $3 $0))))))) (logo_MULL logo_epsL $2) logo_epsA 7 $0)))))
        stitch_uncanonical: (logo_forLoop 7 (lam (lam (fn_0 (logo_MULL logo_epsL #0) logo_epsA 7 $0))))
        stitch_canonical: (logo_forLoop 7 (lam (lam (fn_0 (logo_MULL logo_epsL #0) logo_epsA 7 $0))))
        arity: 1

    """
    invs = deepcopy(start_from) if start_from is not None else []

    assert dc_invs == sorted(dc_invs, key=len), "sort in increasing length so that earlier inventions always come before inventions that use them internally (dreamcoder sometimes varies in output ordering so we must do this)"

    # enumerate in order of increasing length
    for i,dc_string in enumerate(dc_invs):
        # rewrite all nested inventions
        with_sub_inventions = dc_string
        for entry in invs[::-1]: # reverse so we rewrite with larger inventions first to not mangle the internals
            with_sub_inventions = with_sub_inventions.replace(entry['dreamcoder'], entry['name'])
        assert with_sub_inventions.count("#") != 0, f"this no longer seems to be an invention, maybe it was already present in start_from?"
        assert with_sub_inventions.count("#") == 1, f"there seem to still be nested inventions after rewriting, was start_from in sorted order and does dc_invs build on it?\n{with_sub_inventions}\n{dc_invs}\n{start_from}"

        # now that there are no nested inventions, convert the top level to stitch format + canonicalize
        stitch_uncanonical,arity = stitch_format(with_sub_inventions)
        stitch_canonical = canonicalize(stitch_uncanonical)

        invs.append({'name':f'fn_{i}','dreamcoder':dc_string, 'stitch_uncanonical':stitch_uncanonical, 'stitch_canonical':stitch_canonical, 'arity':arity})
    return invs

def to_stitch_program(program: str, stitch_dsl):
    """
    take a dreamcoder program and rewrite it into a stitch program using a stitch dsl, specifically:
    - rewrite #(...) to `fn_i`
    - rewrite lambda -> lam
    """
    # rewrite in reverse order so that subinventions come later so we dont mangle by rewriting within a larger invention
    for entry in stitch_dsl[::-1]:
        program = program.replace(entry['dreamcoder'], entry['name'])
    program = program.replace('lambda','lam')
    assert '#' not in program
    return program

COST_NONTERMINAL = 1
COST_TERMINAL = 100
def stitch_size(stitch_program):
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


def process_dreamcoder_inventions(in_file, out_file):
    # load dreamcoder files and diff
    in_json = load(in_file)
    in_invs_dc = extract_inventions_dreamcoder(in_json)
    out_json = load(out_file)
    out_invs_dc = extract_inventions_dreamcoder(out_json)

    # convert to stitch format 
    in_invs_stitch = to_stitch_invs(in_invs_dc)
    out_invs_stitch = to_stitch_invs(diff_dreamcoder(in_invs_dc, out_invs_dc), start_from=in_invs_stitch)
    new_invs_stitch = diff_stitch(in_invs_stitch, out_invs_stitch)

    in_frontiers_dc = [[p['program'] for p in f['programs']] for f in in_json['frontiers']]
    out_frontiers_dc = [[p['program'] for p in f['programs']] for f in out_json['frontiers']]

    in_frontiers_stitch = [[to_stitch_program(p,in_invs_stitch) for p in ps] for ps in in_frontiers_dc]
    out_frontiers_stitch = [[to_stitch_program(p,out_invs_stitch) for p in ps] for ps in out_frontiers_dc]
    
    # todo add in invention size
    inv_size = sum([stitch_size(inv['stitch_canonical']) for inv in new_invs_stitch],0)

    compression_ratio_min = (sum([min([stitch_size(p) for p in ps]) for ps in in_frontiers_stitch])
    / (sum([min([stitch_size(p) for p in ps]) for ps in out_frontiers_stitch]) + inv_size))


    return {
        'metrics': {
            'absolute_compression': 0,
            'compression_ratio': 0,
            'compression_ratio_min': compression_ratio_min,
        },
        'inventions': new_invs_stitch,
    }


def process_stitch_inventions(in_file, out_file):
    in_json = load(in_file)
    in_invs_dc = extract_inventions_dreamcoder(in_json)
    in_invs_stitch = to_stitch_invs(in_invs_dc)
    out_json = load(out_file)

    new_invs_stitch = stitch_json_to_stitch_invs(out_json)

    in_frontiers_dc = [[p['program'] for p in f['programs']] for f in in_json['frontiers']]
    in_frontiers_stitch = [[to_stitch_program(p,in_invs_stitch) for p in ps] for ps in in_frontiers_dc]

    in_programs_dc = [programs['program'] for f in in_json['frontiers'] for programs in f['programs']]
    in_programs_stitch = [to_stitch_program(p,in_invs_stitch) for p in in_programs_dc]

    out_programs_stitch = out_json["rewritten"]
    out_frontiers_stitch = []
    # todo assuming stitch maintains the order of input programs so we can reconstruct the frontiers as contiguous
    i = 0
    for f in in_frontiers_stitch:
        out_frontiers_stitch.append([])
        for p in f:
            out_frontiers_stitch[-1].append(out_programs_stitch[i])
            i += 1
    assert i == len(out_programs_stitch)

    # todo add in invention size
    inv_size = sum([stitch_size(inv['stitch_canonical']) for inv in new_invs_stitch],0)

    compression_ratio_min = (sum([min([stitch_size(p) for p in ps]) for ps in in_frontiers_stitch])
        / (sum([min([stitch_size(p) for p in ps]) for ps in out_frontiers_stitch]) + inv_size))

    # todo add checks to make sure these are all the same as whats inside out_json

    assert sum([min([stitch_size(p) for p in ps]) for ps in in_frontiers_stitch]) == out_json["original_cost"]

    return {
        'metrics': {
            'absolute_compression': 0,
            'compression_ratio': 0,
            'compression_ratio_min': compression_ratio_min,
        },
        'inventions': new_invs_stitch,
    }



def usages(fn_name, stitch_programs):
    # we count name + closeparen or name + space so that fn_1 doesnt get counted for things like fn_10
    return sum([p.count(f'{fn_name})') + p.count(f'{fn_name} ') for p in stitch_programs])

def diff_dreamcoder(in_invs,out_invs):
    """
    diffs two dreamcoder invention lists and returns their difference. doesnt assume that new inventions
    come at the end. Sorts resulting productions by length.
    inputs to this function shoud look like lists of "#(...)" and so will the returned diff
    """

    assert len(out_invs) - len(in_invs) >= 0, "input invs list is larger than output invs list"

    learned = [inv for inv in out_invs if inv not in in_invs]
    learned.sort(key=len)

    assert len(learned) == len(out_invs) - len(in_invs)

    return learned


def diff_stitch(input_dsl,output_dsl):
    """
    diffs two stitch dsls and returns their difference.
    Assumes (and asserts) that difference comes at the END of the dsls
    """
    difference = len(output_dsl) - len(input_dsl)
    assert difference >= 0
    if difference != 0:
        assert input_dsl == output_dsl[:-difference], f"input:\n{input_dsl}\noutput:\n{output_dsl}"
        return output_dsl[-difference:]
    else:
        assert input_dsl == output_dsl
        return []

def get_benches(bench_dir):
    # assert bench_dir.parent.name == 'benches'
    bench_names = [] # eg ["bench000_iteration_0", ...]
    for file in bench_dir.glob('bench*.json'):
        bench_names.append(file.stem)
    assert len(bench_names) > 0
    bench_names.sort()
    for i,bench in enumerate(bench_names):
        assert bench.startswith(f'bench{i:03d}')
    return bench_names

def latest(mode_dir, benches_dir):
    benches_dir = Path(benches_dir)
    benches = [b for b in benches_dir.iterdir() if b != 'old']
    assert mode_dir in ('dreamcoder','stitch')
    mode_dir = 'stitch' if mode_dir == 'stitch' else 'dc'
    
    res = []
    for benchpath in benches:
        target = benchpath / 'out' / mode_dir
        if not target.exists() or len(list(target.iterdir())) == 0:
            continue # skip if doesnt exist or empty
        runs = [r for r in target.iterdir() if r.is_dir()]
        runs.sort(key=lambda r: r.name) # so last one will be latest by time
        res.append(runs[-1])
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
    
    elif mode == 'to_input_files':
        out_path = Path(sys.argv[2])
        in_files = [x['in_file'] for x in load(out_path / 'invention_info' / 'info.json')]
        for i,in_file in enumerate(in_files):
            print(in_file)
    

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
    
    elif mode == 'process':
        mode = sys.argv[2]
        assert mode in ['stitch','dreamcoder']
        # dir like benches/fake_logo_2019-03-23T18:06:23.106382/out/dc/2021-03-02_00-00-00/
        run_dir = Path(sys.argv[3])
        raw_path = run_dir / 'raw'
        stderr_path = run_dir / 'stderr'
        processed_path = run_dir / 'processed'
        if not raw_path.exists():
            print("Can't find raw/ directory, did you provide a path like benches/bench_name/out/dc/2021-03-02_00-00-00/ ?")
        if not processed_path.exists():
            processed_path.mkdir()
        bench_dir = run_dir.parent.parent.parent
        bench_group = bench_dir.name # eg fake_logo_2019-03-23T18:06:23.106382
        bench_names = get_benches(bench_dir)
        
        for bench in bench_names:
            if not (raw_path / f'{bench}.json').exists() or os.stat(raw_path / f'{bench}.json').st_size == 0:
                continue # this bench was missing or empty
            print(f'processing {bench}')

            processed = {
                'bench_group': bench_group,
                'bench': bench,
                'mode': mode,
                'run': str(run_dir),
                'metrics': {
                    'time_binary_seconds': None,
                    'time_per_inv_with_rewrite': None,
                    'time_per_inv_no_rewrite': None,
                    'mem_peak_kb': None,
                    'compression_ratio': None,
                    'absolute_compression': None
                },
                'num_inventions': None,
                'inventions': {},
            }

            raw = load(raw_path / f'{bench}.json')
            with open(stderr_path / f'{bench}.stderr') as f:
                stderr = f.read().split('\n')

            # time_binary_seconds: wall clock on the whole binary running - not super precise but okay
            [x] = [l for l in stderr if 'Elapsed (wall clock) time' in l]
            x = x.split(' ')[-1]
            if x.count(':') == 1:
                # m:s format
                mins,secs = x.split(':')
                processed['metrics']['time_binary_seconds'] = int(mins) * 60 + float(secs)
            elif x.count(':') == 2:
                # h:m:s format
                hours,mins,secs = x.split(':')
                processed['metrics']['time_binary_seconds'] = int(hours) * 3600 + int(mins) * 60 + float(secs)
            else:
                assert False
            
            # mem_peak_kb: memory use
            [x] = [l for l in stderr if 'Maximum resident set size (kbytes)' in l]
            processed['metrics']['mem_peak_kb'] = int(x.split(' ')[-1])

            # process the output dsls to get inventions and scores
            if mode == "dreamcoder":
                res = process_dreamcoder_inventions(bench_dir / f'{bench}.json', raw_path / f'{bench}.json')
            elif mode == "stitch":
                res = process_stitch_inventions(bench_dir / f'{bench}.json', raw_path / f'{bench}.json')
            else:
                assert False
            
            for k,v in res['metrics'].items():
                processed['metrics'][k] = v
            
            processed['inventions'] = res['inventions']
            processed['num_inventions'] = len(processed['inventions'])

            def get_times(line_prefix):
                return [int(line.split(':')[1].strip()) for line in stderr if line.startswith(line_prefix)]
            
            # time_per_inv_with_rewrite and time_per_inv_no_rewrite
            if mode == 'dreamcoder':
                no_rewrite = get_times('Timing Comparison Point A (vspace+beam) (millis):')
                with_rewrite = get_times('Timing Comparison Point B (vspace+beam+batched_rewrite) (millis):')
                if len(no_rewrite) == processed['num_inventions'] + 1:
                    no_rewrite = no_rewrite[:-1] # drop the last one which was rejected
                if len(with_rewrite) == processed['num_inventions'] + 1:
                    with_rewrite = with_rewrite[:-1] # drop the last one which was rejected

            elif mode == 'stitch':
                no_rewrite = get_times('Timing Comparison Point A (search) (millis):')
                with_rewrite = get_times('Timing Comparison Point B (search+rewrite) (millis):')
            
            assert len(no_rewrite) == processed['num_inventions'], f"{len(no_rewrite)} != {processed['num_inventions']}"
            assert len(with_rewrite) == processed['num_inventions'], f"{len(with_rewrite)} != {processed['num_inventions']}"

            if processed['num_inventions'] != 0:
                processed['metrics']['time_per_inv_with_rewrite'] = sum(with_rewrite) / len(with_rewrite)
                processed['metrics']['time_per_inv_no_rewrite'] = sum(no_rewrite) / len(no_rewrite)
            else:
                processed['metrics']['time_per_inv_with_rewrite'] = 0
                processed['metrics']['time_per_inv_no_rewrite'] = 0

            save(processed, processed_path / f'{bench}.json')
    elif mode == 'iteration_budget':
        """
        analyze.py iteration_budget <dreamcoder dir to compare to> <specific benchmark json to compare on>
        or compare against "none" for 20 iterations automatically
        """
        if sys.argv[2] == 'none':
            print(20)
            sys.exit(0)
        compare_to = Path(sys.argv[2])
        bench = Path(sys.argv[3])
        assert str(bench).endswith('.json')
        if not (compare_to / 'processed' / bench.name).exists():
            print(0)
            sys.exit(0)
            # print(f"Cant find {compare_to}/processed/{bench.name}; run with `loose` to set iter budget to 0")
        num_invs = load(compare_to / 'processed' / bench.name)['num_inventions']
        print(num_invs)
        sys.exit(0)
    elif mode == 'graphs':
        """
        python3 analyze.py graphs bar <run dir 1> <run dir 2> <run dir 3> ...
        where a run_dir is a timestamped dir like benches/fake_logo_2019-03-23T18:06:23.106382/out/dc/2021-03-02_00-00-00/
        and alternatively use "line" instead of "bar"
        """
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42




        type = sys.argv[2]
        assert type in ('bar','line')
        run_dirs = [Path(x).absolute() for x in sys.argv[3:]]

        bench_dir = run_dirs[0].parent.parent.parent
        assert all([str(x.parent.parent.parent) == str(bench_dir) for x in run_dirs]), "not all came from same benchmark group"
        bench_group = bench_dir.name # eg fake_logo_2019-03-23T18:06:23.106382
        bench_names = get_benches(bench_dir)

        # pool all metrics that are non-none for at least one run of one benchmark
        metrics = []
        for bench in bench_names:
            for run_dir in run_dirs:
                if not (run_dir / 'processed' / f'{bench}.json').exists():
                    continue
                processed = load(run_dir / 'processed' / f'{bench}.json')
                metrics.extend([metric for metric,val in processed['metrics'].items() if val is not None])
        metrics = sorted(list(set(metrics)))


        num_runs = len(run_dirs)
        num_benches = len(bench_names)
        bar_width = (1/(num_runs+1))

        MEMORY = 'mem_peak_kb'
        TIME_BINARY = 'time_binary_seconds'
        TIME_TOTAL = 'time_total_seconds'
        TIME_CANDIDATES = 'time_candidates_seconds'
        TIME_MIDDLE_REWRITE = 'time_middle_rewrite_seconds'
        TIME_FINAL_REWRITE = 'time_final_rewrite_seconds'
        COMPRESSION_RATIO = 'compression_ratio'
        COMPRESSION_RATIO_MIN = 'compression_ratio_min'
        COMPRESSION_UTILITY = 'compression_utility'
        STITCH_UTILITY = 'stitch_utility'
        PCFG_SCORE = 'pcfg_score'

        # for each metric, make a bar graph with the bench name on the
        # x axis and the metric value on the y axis, with one bar per run
        for metric in metrics:
            plt.clf()
            fig,ax = plt.subplots()
            for i,run_dir in enumerate(run_dirs):
                bar_heights = []
                for bench in bench_names:
                    if not (run_dir / 'processed' / f'{bench}.json').exists():
                        bar_heights.append(0)
                    else:
                        bar_heights.append(load(run_dir / 'processed' / f'{bench}.json')['metrics'][metric])
                
                xs = [j + i * bar_width for j in range(num_benches)]
                
                ax.bar(xs, bar_heights, width = bar_width, label=f'{run_dir.parent.name}/{run_dir.name}')

            # set y axis to start at 1
            if metric in (COMPRESSION_RATIO,COMPRESSION_RATIO_MIN):
                ax.set_ylim(bottom=1)
            
            if metric in (MEMORY,TIME_BINARY):
                ax.set_yscale('log')

            plt.xlabel('benchmark')
            plt.ylabel(metric)
            plt.title(f'{metric} {bench_group}')
            xs = [j + (bar_width*(num_runs-1))/2 for j in range(num_benches)]
            plt.xticks(xs, bench_names, rotation=90, fontsize=8)
            plt.legend()
            plt.tight_layout()

            os.makedirs(bench_dir / 'plots', exist_ok=True)
            plt.savefig(bench_dir / 'plots' / f'{metric}_{bench_group}.png',dpi=400)
            print("wrote to " + str(bench_dir / 'plots' / f'{metric}_{bench_group}.png'))
            

    elif mode == 'artifact_to_bench':
        """
        convert from dirs that ./extract_all_data.sh outputs to a benches/ benchmark.
        """

        for domain in ('logo','regex'):
            flat_bench_group = Path('benches') / f'{domain}_all'
            flat_bench_group.mkdir(exist_ok=True)
            flat_i = 0
            for run_i,run in enumerate(RUNS[domain]):
                assert (Path('data') / domain / run / 'iteration_0.json').exists()
                bench_group = Path('benches') / f'{domain}_{run}'
                bench_group.mkdir(exist_ok=True)
                for i in range(20):
                    iteration_json = Path('data') / domain / run / f'iteration_{i}.json'
                    if not iteration_json.exists():
                        break
                    shutil.copy(iteration_json, bench_group / f'bench{i:03d}_it{i}.json')
                    shutil.copy(iteration_json, flat_bench_group / f'bench{flat_i:03d}_it{i}_run{run_i}.json')
                    flat_i += 1

    elif mode == 'artifact_rec-fp':
        source = Path('data/rec-fp/iteration=10_2019-07-11T19:49:10.899159')
        bench_group_arity4 = Path('benches/rec-fp_arity4_2019-07-11T19:49:10.899159')
        bench_group_arity3 = Path('benches/rec-fp_arity3_2019-07-11T19:49:10.899159')
        bench_group_arity4.mkdir(exist_ok=True)
        bench_group_arity3.mkdir(exist_ok=True)
        for i in range(20):
            iteration_json = source / f'iteration_{i}.json'
            if not iteration_json.exists():
                break
            shutil.copy(iteration_json, bench_group_arity4 / f'bench{i:03d}_it{i}.json')
            d = load(iteration_json)
            d['arity'] = 3
            save(d, bench_group_arity3 / f'bench{i:03d}_it{i}.json')
            
    elif mode == 'latest':

        for l in latest(sys.argv[2],sys.argv[3]):
            print(str(l))

    # elif mode == 'claim-1':
    #     benches_dir = sys.argv[2]
    #     runs = latest('dreamcoder',benches_dir)
    #     for run in runs:
    #         run = run.parent.parent.parent
    #         out_dir=f"{benches_dir}/out/stitch/$(TZ='America/New_York' date '+%Y-%m-%d_%H-%M-%S')"
    #         mkdir -p $OUT_DIR/raw
    #         mkdir -p $OUT_DIR/stderr
    #         STITCH_FLAGS="--max-arity=3 --threads=1 --fmt=dreamcoder --dreamcoder-comparison"
    #         for BENCH_PATH in $BENCH_DIR/bench*.json; do
    #             BENCH=$(basename -s .json $BENCH_PATH)
    #             ITERATIONS=$(python3 analyze.py iteration_budget $COMPARE_TO $BENCH_PATH)
    #             if [ $ITERATIONS -eq 0 ]; then
    #                 echo "skipping $BENCH_PATH since comparison is nonexistant or has zero iterations"
    #                 continue
    #             fi
    #             echo "[bench_stitch.sh] Running Stitch on: $BENCH"
    #             echo "$STITCH_DIR/target/release/compress $BENCH_PATH --iterations=$ITERATIONS $STITCH_FLAGS $EXTRA_STITCH_FLAGS"
    #             $GTIME -v $STITCH_DIR/target/release/compress $BENCH_PATH --iterations=$ITERATIONS $STITCH_FLAGS $EXTRA_STITCH_FLAGS --out=$OUT_DIR/raw/$BENCH.json 2>&1 &> $OUT_DIR/stderr/$BENCH.stderr
    #         done

    #         python3 analyze.py process stitch $OUT_DIR

    #         echo "Done: $OUT_DIR"

    elif mode == 'graph_all':
        """
        python3 analyze.py graph_all benches/
        """
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        import numpy as np
        import seaborn as sns
        sns.set_theme(color_codes=True)

        benches_dir = Path(sys.argv[2])


        # get all the data

        data_by_domain = {}
        domains = ['logo', 'list', 'towers', 'text', 'physics']
        metrics = ["time_per_inv_with_rewrite", "time_per_inv_no_rewrite", "mem_peak_kb", "compression_ratio_min"]


        for domain in domains:
            total_benches = 0
            done_benches = 0
            zero_inventions = 0
            skipped = []
            stitch_metrics = {
                "time_per_inv_with_rewrite": [],
                "time_per_inv_no_rewrite": [],
                "mem_peak_kb": [],
                "compression_ratio": [],
                "compression_ratio_min": [],
            }
            dreamcoder_metrics = {
                "time_per_inv_with_rewrite": [],
                "time_per_inv_no_rewrite": [],
                "mem_peak_kb": [],
                "compression_ratio": [],
                "compression_ratio_min": [],
            }
            benchgroups = [bg for bg in os.listdir(benches_dir) if bg.startswith(domain)]
            for benchgroup in benchgroups:
                benches = [b for b in os.listdir(benches_dir/benchgroup) if b.endswith('.json') and b.startswith('bench')]
                total_benches += len(benches)
                stitch_dir = benches_dir / benchgroup / 'out' / 'stitch'
                dreamcoder_dir = benches_dir / benchgroup / 'out' / 'dc'
                if not stitch_dir.exists() or not dreamcoder_dir.exists() or len(list(stitch_dir.iterdir())) == 0 or len(list(dreamcoder_dir.iterdir())) == 0:
                    skipped.append(f'GROUP: {benchgroup}')
                    continue

                # get the most recent file
                stitch_run = sorted(list(stitch_dir.iterdir()),key=lambda x: x.name)[-1]
                dreamcoder_run = sorted(list(dreamcoder_dir.iterdir()),key=lambda x: x.name)[-1]

                for bench in benches:
                    stitch_bench = stitch_run / 'processed' / bench
                    dreamcoder_bench = dreamcoder_run / 'processed' / bench
                    if not stitch_bench.exists() or not dreamcoder_bench.exists():
                        skipped.append(f'BENCH: {benchgroup}/{bench}')
                        continue
                    
                    stitch_processed = load(stitch_bench)
                    dreamcoder_processed = load(dreamcoder_bench)
                    assert stitch_processed['num_inventions'] == dreamcoder_processed['num_inventions']
                    num_inventions = stitch_processed['num_inventions']

                    # we dont record metrics on runs that have no inventions
                    if num_inventions == 0:
                        done_benches += 1
                        zero_inventions += 1
                        continue

                    for metric in metrics:
                        stitch_data = stitch_processed["metrics"][metric]
                        dreamcoder_data = dreamcoder_processed["metrics"][metric]


                        if metric in ("time_per_inv_with_rewrite", "time_per_inv_no_rewrite"):
                            # scale from ms -> s
                            stitch_data /= 1000
                            dreamcoder_data /= 1000

                        if metric in ('compression_ratio','compression_ratio_min'):
                            stitch_data, dreamcoder_data = stitch_data / dreamcoder_data, dreamcoder_data / stitch_data
                        stitch_metrics[metric].append(stitch_data)
                        dreamcoder_metrics[metric].append(dreamcoder_data)
                    done_benches += 1
            
            data_by_domain[domain] = {
                'stitch': stitch_metrics,
                'dreamcoder': dreamcoder_metrics,
                'skipped': skipped,
                'total_benches': total_benches,
                'done_benches': done_benches,
                'zero_inventions': zero_inventions,
            }

        print("Gathered all data, on to graphing...")

        for domain,data in data_by_domain.items():
            done = data['done_benches']
            total = data['total_benches']
            zero = data['zero_inventions']
            print(f"Found {domain} data: {done}/{total}; zero: {zero}; skipped:")
            for skipped in data['skipped']:
                print(f"\n{skipped}")
            if domain == 'text':
                print("Note: We expect the following to be skipped as DreamCoder produces zero abstractions for them:")
                print("\ttext_text_ellisk_2019-01-24T21.58.02/bench010_it12.json")
                print("\ttext_text_ellisk_2019-01-25T20.19.06/bench006_it6.json")

        num_x_ticks = len(domains)
        num_bars_per_tick = 2
        bar_width = (1/(num_bars_per_tick+1))


        # for each metric, make a bar graph with the bench name on the
        # x axis and the metric value on the y axis, with one bar per run
        for metric in metrics:
            plt.clf()
            fig,ax = plt.subplots()

            if metric not in ('compression_ratio','compression_ratio_min'):
                for i,name in enumerate(['stitch','dreamcoder']):
                    ys = []
                    y_errbars = []
                    xs = []
                    for j,domain in enumerate(domains):
                        ydata = data_by_domain[domain][name][metric]
                        if len(ydata) == 0:
                            continue
                        ys.append(sum(ydata)/len(ydata))
                        y_errbars.append(np.std(ydata))
                        xs.append(j + i*bar_width)
                    
                    label = {'stitch':'Stitch (1 cpu)','dreamcoder':'Dreamcoder (8 cpus)'}[name]
                    ax.bar(xs, ys, yerr=y_errbars, width = bar_width, bottom=0, label=label)
                xs = [j + bar_width/2 for j in range(len(domains))]
                xticks = [{
                    'logo':'LOGO',
                    'towers':'Towers',
                    'list': 'Lists',
                    'text': 'Text',
                    'physics': 'Physics',
                }[name] for name in domains]

                plt.xticks(xs, xticks)
            else:
                data = []
                included_domains = []
                for j,domain in enumerate(domains):
                    ydata = data_by_domain[domain]['stitch'][metric]
                    if len(ydata) > 0:
                        data.append(ydata)
                        included_domains.append(domain)
                if len(data) != 0:
                    ax.violinplot(data)
                    plt.axhline(y=1, color='b', linewidth=1, linestyle='dashed')

                    xticks = [{
                        'logo':'LOGO',
                        'towers':'Towers',
                        'list': 'Lists',
                        'text': 'Text',
                        'physics': 'Physics',
                    }[name] for name in included_domains]

                    plt.xticks(list(range(1,len(xticks)+1)), xticks)


            # set y axis to start at 1
            if metric in ('time_per_inv_with_rewrite', 'time_per_inv_no_rewrite'):
                ax.set_ylim(bottom=.001,top=1000)
            if metric == 'mem_peak_kb':
                ax.set_ylim(bottom=1000)
            
            if metric in ('time_per_inv_with_rewrite','time_per_inv_no_rewrite', 'mem_peak_kb'):
                ax.set_yscale('log')
                plt.grid(True, which='minor', linewidth=1, alpha=.5)
                # plt.grid(True, which='major', linewidth=1, color='gray')



            plt.xlabel('Domain')
            if metric.startswith('compression_ratio'):
                plt.title('Rewritten corpus size ratio (DreamCoder / Stitch)')
            else:
                plt.ylabel(
                    {
                        'time_per_inv_with_rewrite': 'Time per abstraction (seconds)',
                        'time_per_inv_no_rewrite': 'Time per abstraction (s) (no rewriting)',
                        'mem_peak_kb': 'Peak memory use (KB)',
                    }[metric])
            # plt.ylabel(f'{metric}')
            plt.legend()
            plt.tight_layout()

            os.makedirs('plots', exist_ok=True)
            plt.savefig(f'plots/{benches_dir.name}_{metric}.png',dpi=400)
            plt.savefig(f'plots/{benches_dir.name}_{metric}.pdf')
            print(f"wrote to plots/{benches_dir.name}_{metric}.png and pdf")


    elif mode == 'show_usages':
        stderr = Path(sys.argv[2])

        with open(stderr) as f:
            lines = f.read().split('\n')

        i = 0
        invs = 0
        while True:
            if not lines[i].startswith('Improved score to'):
                i += 1
                if i >= len(lines):
                    break
                continue
            # expect next line is like "#(lambda (lambda (lambda (fix1 $2 (lambda (lambda (if (empty? $0) $2 ($3 ($1 (cdr $0)) (car $0))))))))) : list(t0) -> (t1 -> t0 -> t1) -> t1 -> t1"
            i += 1
            invention = lines[i].split(':')[0].strip()
            print(f'Invention fn_{invs}: {invention}')
            while not lines[i].startswith('New primitive is used'):
                i += 1
                continue # internal continue to this inner while loop
            num_usages = int(lines[i].split(' ')[4])
            print(f'usages: {num_usages}')
            i += 1
            assert lines[i].startswith('Here is where it is used')
            i += 1
            j = 0
            while lines[i].strip().startswith('('):
                assert invention in lines[i]
                usage = lines[i].strip().replace(invention, f'fn_{invs}')
                print(usage)
                i += 1
                j += 1
            assert j == num_usages
            invs += 1
            print()
    elif mode == 'count':
        from glob import glob
        total = 0
        count = 0
        m = 0
        for bench in glob('benches/*/out/stitch/*/stderr/*'):
            with open(bench,'r') as f:
                for line in f.readlines():
                    if 'max depth:' in line:
                        n = int(line.split(' ')[-1])
                        total += n
                        count += 1
                        if n > m:
                            print(n)
                            m = n
            print("Avg",total/count)
    else:
        assert False, f"mode not recognized: {mode}"







    

