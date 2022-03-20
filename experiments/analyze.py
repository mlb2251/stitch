import json
import sys
import os

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


if len(sys.argv) != 3:
    print("Usage: python3 analyze.py INFILE OUTFILE")
    sys.exit(1)

with open(sys.argv[1],'rb') as f:
    data = json.load(f)



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
dsl_dc = [prod["expression"] for prod in data["DSL"]["productions"]]
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
    
# for d in dsl:
#     for k,v in d.items():
#         print(f'{k}: {v}')
#     print()

with open(sys.argv[2],'w') as f:
    data = json.dump(dsl,f,indent=4)
# print(f"wrote {sys.argv[2]}")