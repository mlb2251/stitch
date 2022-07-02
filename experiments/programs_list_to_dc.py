import json
import sys
from pathlib import Path

if __name__ == '__main__':
    with  open(sys.argv[1],'r') as f:
        programs = json.load(f)
    assert isinstance(programs,list)
    programs = [p.replace('lam ','lambda ') for p in programs]
    for i in ["1", "2", "3", "4", "5", "6", "8", "9", "0", "7", "pi", "-1", "-", "+", "*", "repeat", "r_s"]:
        programs = [p.replace(f' {i} ', f' zzz{i} ') for p in programs]
        programs = [p.replace(f' {i} ', f' zzz{i} ') for p in programs]
        programs = [p.replace(f'({i} ', f'(zzz{i} ') for p in programs]
        programs = [p.replace(f' {i})', f' zzz{i})') for p in programs]


    message = {
        "no_stopping_criterion": True,
        "verbose": False,
        "arity": 1,
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
            "productions": [
                {"expression": "zzzpi", "logProbability": 0.0 },
                {"expression": "0.5", "logProbability": 0.0 },
                {"expression": "0.75", "logProbability": 0.0 },
                {"expression": "1.5", "logProbability": 0.0 },
                {"expression": "zzz1", "logProbability": 0.0 },
                {"expression": "1.25", "logProbability": 0.0 },
                {"expression": "1.75", "logProbability": 0.0 },
                {"expression": "zzz2", "logProbability": 0.0 },
                {"expression": "2.25", "logProbability": 0.0 },
                {"expression": "2.5", "logProbability": 0.0 },
                {"expression": "2.75", "logProbability": 0.0 },
                {"expression": "zzz3", "logProbability": 0.0 },
                {"expression": "3.25", "logProbability": 0.0 },
                {"expression": "3.5", "logProbability": 0.0 },
                {"expression": "zzz4", "logProbability": 0.0 },
                {"expression": "zzz5", "logProbability": 0.0 },
                {"expression": "zzz6", "logProbability": 0.0 },
                {"expression": "3.75", "logProbability": 0.0 },
                {"expression": "zzz8", "logProbability": 0.0 },
                {"expression": "zzz9", "logProbability": 0.0 },
                {"expression": "10", "logProbability": 0.0 },
                {"expression": "11", "logProbability": 0.0 },
                {"expression": "4.25", "logProbability": 0.0 },
                {"expression": "4.5", "logProbability": 0.0 },
                {"expression": "4.75", "logProbability": 0.0 },
                {"expression": "5.25", "logProbability": 0.0 },
                {"expression": "5.5", "logProbability": 0.0 },
                {"expression": "5.75", "logProbability": 0.0 },
                {"expression": "zzz0", "logProbability": 0.0 },
                {"expression": "0.25", "logProbability": 0.0 },
                {"expression": "6.25", "logProbability": 0.0 },
                {"expression": "6.5", "logProbability": 0.0 },
                {"expression": "6.75", "logProbability": 0.0 },
                {"expression": "-0.75", "logProbability": 0.0 },
                {"expression": "-0.5", "logProbability": 0.0 },
                {"expression": "zzz7", "logProbability": 0.0 },
                {"expression": "7.25", "logProbability": 0.0 },
                {"expression": "7.5", "logProbability": 0.0 },
                {"expression": "7.75", "logProbability": 0.0 },
                {"expression": "-0.25", "logProbability": 0.0 },
                {"expression": "8.25", "logProbability": 0.0 },
                {"expression": "8.5", "logProbability": 0.0 },
                {"expression": "8.75", "logProbability": 0.0 },
                {"expression": "9.25", "logProbability": 0.0 },
                {"expression": "9.5", "logProbability": 0.0 },
                {"expression": "9.75", "logProbability": 0.0 },
                {"expression": "12", "logProbability": 0.0 },
                {"expression": "-1.25", "logProbability": 0.0 },
                {"expression": "-2.5", "logProbability": 0.0 },
                {"expression": "-2.25", "logProbability": 0.0 },
                {"expression": "-2", "logProbability": 0.0 },
                {"expression": "zzz-1", "logProbability": 0.0 },
                {"expression": "-1.5", "logProbability": 0.0 },
                {"expression": "-3", "logProbability": 0.0 },
                {"expression": "-2.75", "logProbability": 0.0 },
                {"expression": "-1.75", "logProbability": 0.0 },
                {"expression": "zzz-", "logProbability": 0.0 },
                {"expression": "zzz+", "logProbability": 0.0 },
                {"expression": "zzz*", "logProbability": 0.0 },
                {"expression": "/", "logProbability": 0.0 },
                {"expression": "pow", "logProbability": 0.0 },
                {"expression": "sin", "logProbability": 0.0 },
                {"expression": "cos", "logProbability": 0.0 },
                {"expression": "tan", "logProbability": 0.0 },
                {"expression": "max", "logProbability": 0.0 },
                {"expression": "min", "logProbability": 0.0 },
                {"expression": "M", "logProbability": 0.0 },
                {"expression": "T", "logProbability": 0.0 },
                {"expression": "C", "logProbability": 0.0 },
                {"expression": "zzzrepeat", "logProbability": 0.0 },
                {"expression": "empt", "logProbability": 0.0 },
                {"expression": "l", "logProbability": 0.0 },
                {"expression": "c", "logProbability": 0.0 },
                {"expression": "r", "logProbability": 0.0 },
                {"expression": "zzzr_s", "logProbability": 0.0 }
            ]
        },
        "iterations": 10,
        "frontiers": [
            {
                "request": {
                    "constructor": "tstroke",
                    "arguments": []
                },
                "programs": [
                    {
                        "program": p,
                        "logLikelihood": 0.0
                    },
                ]
            }
            for p in programs]
    }

    with open(Path(sys.argv[1]).with_suffix('.dc.json'),'w') as f:
        programs = json.dump(message, f, indent=4)


