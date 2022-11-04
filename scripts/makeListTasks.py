"""
This is a modification of https://github.com/mlb2251/stitch_dreamcoder/blob/master/dreamcoder/domains/list/makeListTasks.py
to generate the origami list tasks in a way 

"""

from random import seed,random,randint
import json

class Task():
    def __init__(self, name, tp, ios):
        print(f"Task {name} :: {tp}")
        for (inputs,output) in ios:
            print(f"\t{inputs} -> {output}")
        self.name = name
        self.tp = tp
        self.ios = ios

def tlist(x):
    return f"list ({x})"
def arrow(*args):
    return "(" + " -> ".join(args) + ")"

tint = "int"
tbool = "bool"



def make_list_bootstrap_tasks():
    seed(42)

    def suffixes(l):
        if l == []:
            return []
        else:
            return [l[1:]] + suffixes(l[1:])

    def flip(): return random() > 0.5

    def randomSuffix():
        return [randint(0, 9) for _ in range(randint(1, 4))]

    def randomList(minimum=0, minimumLength=4, maximumLength=6):
        return [randint(minimum, 9) for _ in range(randint(minimumLength, maximumLength))]

    def randomListOfLists():
        return [randomSuffix() for _ in range(randint(2, 4))]

    def randomListOfLists_bool(l=None):
        if l is None:
            l = randint(4, 7)
        return [randomBooleanList() for _ in range(l)]

    def randomBooleanList():
        return [flip() for _ in range(randint(4, 7))]

    # Reliably learned in under a minute; always triggers learning of length
    # primitive
    lengthBootstrap = [
        Task("length int", arrow(tlist(tint), tint),
             [((l,), len(l))
              for _ in range(10)
              for l in [randomList()]]),
        Task("map length", arrow(tlist(tlist(tint)), tlist(tint)),
             [((xss,), [len(xs) for xs in xss])
              for _ in range(10)
              for xss in [randomListOfLists()] ])
    ]

    # Encourages learning of unfolding
    unfoldBootstrap = [
        Task("countdown", arrow(tint, tlist(tint)),
             [((n,), list(range(n + 1, 1, -1)))
              for n in range(10)]),
        Task("weird count", arrow(tint, tlist(tint)),
             [((n,), list(range(-n,0,-1)))
              for n in range(-10,0) ]),
        Task("take every other", arrow(tlist(tint),tlist(tint)),
             [((l,), [x for j,x in enumerate(l) if j%2 == 0])
              for _ in range(9)
              for l in [ [randint(0, 9) for _ in range(randint(1,4)*2)] ] ] + [(([],),[])]),
        Task("drop last element", arrow(tlist(tint),tlist(tint)),
             [((l,), l[:-1])
              for _ in range(10)
              for l in [ [randint(0, 9) for _ in range(randint(2,5))] ] ]),
        Task("range", arrow(tint, tlist(tint)),
             [((n,), list(range(n)))
              for n in range(10)]),
        Task("range inclusive", arrow(tint, tlist(tint)),
             [((n,), list(range(n + 1)))
              for n in range(10)]),
    ]

    # Encourages learning how to treat a list as an array
    arrayBootstrap = [
        Task("index int", arrow(tint, tlist(tint), tint),
             [((n, l), l[n])
              for n in range(10)
              for l in [[randint(0, 9) for _ in range(randint(n + 1, n + 5))]]]),
        Task("1-index int", arrow(tint, tlist(tint), tint),
             [((n, l), l[n - 1])
              for n in range(1,11)
              for l in [[randint(0, 9) for _ in range(randint(n + 1, n + 4))]]])
    ]


    # learning to fold
    foldBootstrap = [
        Task("stutter", arrow(tlist(tint),tlist(tint)),
             [((l,), [z for x in l for z in [x,x] ])
              for _ in range(10)
              for l in [randomList()] ]),
        Task("sum", arrow(tlist(tint), tint),
             [((l,), sum(l))
              for _ in range(10)
              for l in [randomList()]]),
        Task("append constant 0", arrow(tlist(tint),tlist(tint)),
             [((l,),l + [0])
              for _ in range(10)
              for l in [randomList()] ]),
    ]

    # learning to map
    mapBootstrap = [
        Task("map double", arrow(tlist(tint), tlist(tint)),
             [((l,), list(map(lambda n: n * 2, l)))
              for _ in range(10)
              for l in [randomList()]]),
        Task("map increment", arrow(tlist(tint),tlist(tint)),
             [((l,),list(map(lambda n: n+1, l)))
              for _ in range(10)
              for l in [randomList()] ]),
        Task("map negation", arrow(tlist(tint),tlist(tint)),
             [((l,),list(map(lambda n: 0-n, l)))
              for _ in range(10)
              for l in [randomList()] ]),

    ]

    # Learning to zip lists together
    zipBootstrap = [
        Task("zip plus", arrow(tlist(tint),tlist(tint),tlist(tint)),
             [((l1,l2),list(map(lambda x,y: x+y,l1,l2)))
              for _ in range(10)
              for l1 in [randomList(minimumLength=2, maximumLength=4)]
              for l2 in [[ randint(0,9) for _ in range(len(l1)) ]]]),
        Task("zip minus", arrow(tlist(tint),tlist(tint),tlist(tint)),
             [((l1,l2),list(map(lambda x,y: x-y,l1,l2)))
              for _ in range(10)
              for l1 in [randomList(minimumLength=2, maximumLength=4)]
              for l2 in [[ randint(0,9) for _ in range(len(l1)) ]]]),
    ]

    # Learning to filter
    filterBootstrap = [
        Task("remove 0s",
             arrow(tlist(tint), tlist(tint)),
             [((xs,), [x for x in xs if x != 0])
              for _ in range(10)
              for xs in [[randint(0, 3) for _ in range(5)]]]),
        Task("remove non-positives", # note that this is misnamed and actually keeps entries that are <= 1, and also negatives are never included in the first place by the input sampler
             arrow(tlist(tint), tlist(tint)),
             [((xs,), [x for x in xs if not (x > 1)])
              for _ in range(10)
              for xs in [[randint(0, 3) for _ in range(5)]]]),
    ]

    return lengthBootstrap + filterBootstrap + \
        unfoldBootstrap + arrayBootstrap + foldBootstrap + mapBootstrap + zipBootstrap


tasks = make_list_bootstrap_tasks()

file = "data/synth/origami.json"
with open(file,"w") as f:
    json.dump([{
        "name": task.name,
        "tp": task.tp,
        "ios": [[[str(i) for i in inputs],str(output)] for inputs, output in task.ios]
    } for task in tasks], f, indent=4)
print("wrote data/synth/origami.json")