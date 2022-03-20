import json
import sys
import os
import re

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
# for d in dsl:
#     for k,v in d.items():
#         print(f'{k}: {v}')
#     print()





def diff(input_dsl,output_dsl):
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



