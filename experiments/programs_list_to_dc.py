import json
import sys
from pathlib import Path

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python programs_list_to_dc.py input_file output_file arity graphics|towers")
        print("eg: python programs_list_to_dc.py nuts-bolts.json nuts-bolts.dc.json 3 graphics")
        sys.exit(1)
    with open(sys.argv[1],'r') as f:
        programs = json.load(f)
    assert isinstance(programs,list)
    programs = [p.replace('lam ','lambda ') for p in programs]

    domain = sys.argv[4]


    if domain == 'graphics':
        # prims = ["1", "2", "3", "4", "5", "6", "8", "9", "0", "7", "pi", "-1", "-", "+", "*", "repeat", "r_s", "0.5", ]
        prims = ["pi", "0.5", "0.75", "1.5", "1", "1.25", "1.75", "2", "2.25", "2.5", "2.75", "3", "3.25", "3.5", "4", "5", "6", "3.75", "8", "9", "10", "11", "4.25", "4.5", "4.75", "5.25", "5.5", "5.75", "0", "0.25", "6.25", "6.5", "6.75", "-0.75", "-0.5", "7", "7.25", "7.5", "7.75", "-0.25", "8.25", "8.5", "8.75", "9.25", "9.5", "9.75", "12", "-1.25", "-2.5", "-2.25", "-2", "-1", "-1.5", "-3", "-2.75", "-1.75", "-", "+", "*", "/", "pow", "sin", "cos", "tan", "max", "min", "M", "T", "C", "repeat", "empt", "l", "c", "r", "r_s"]
        request = {
                    "constructor": "tstroke",
                    "arguments": []
                }
    elif domain == 'towers':
        prims = ["t", "h", "r", "l"] + [str(i) for i in range(1,15)]
        request = {
                    "constructor": "->",
                    "arguments": [
                        {
                            "constructor": "ttowersstate",
                            "arguments": []
                        },
                        {
                            "constructor": "ttowersstate",
                            "arguments": []
                        }
                    ]
                }
    else:
        assert False

    for i in prims:
        programs = [p.replace(f' {i} ', f' {domain}_{i} ') for p in programs]
        programs = [p.replace(f' {i} ', f' {domain}_{i} ') for p in programs]
        programs = [p.replace(f'({i} ', f'({domain}_{i} ') for p in programs]
        programs = [p.replace(f' {i})', f' {domain}_{i})') for p in programs]


    message = {
        "no_stopping_criterion": True,
        "verbose": False,
        "arity": int(sys.argv[3]),
        "topK": 2,
        "pseudoCounts": 30.0,
        "aic": 1.0,
        "bs": 1000000,
        "topI": 300,
        "structurePenalty": 1.0,
        "CPUs": 8,
        "lc_score": 0.0,
        "DSL": {
            "logVariable": 0.0,
            "productions": [ {"expression": f"{domain}_{p}", "logProbability": 0.0 } for p in prims ]
        },
        "iterations": 10,
        "frontiers": [
            {
                "request": request,
                "programs": [
                    {
                        "program": f'(lambda ({p[1:-1]} $0))' if domain == 'towers' else p,
                        "logLikelihood": 0.0
                    },
                ]
            }
            for p in programs]
    }

    with open(Path(sys.argv[2]),'w') as f:
        programs = json.dump(message, f, indent=4)


