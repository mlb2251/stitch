import sys
import os
import re

import matplotlib.pyplot as plt 
import numpy as np
from prettytable import PrettyTable

import seaborn as sns
sns.set_theme(color_codes=True)
sns.set(font_scale=2)

def get_stats(path):
    with open(path, 'r') as infile:
        ss = {}
        for line in infile:
            if line.strip():
                ss[str(re.match(r'^([^:]*):', line).group(1))] = int(re.match(r'.*Median: ([^$]*)$', line).group(1))
        return ss

def get_data(path, mode):
    if mode == 'claim-2':
        with open(path, 'r') as infile:
            for line in infile:
                if line.startswith('Cost Improvement: '):
                    train_ratio = float(re.match(r'Cost Improvement: \(([^x]*)x.*', line).group(1))
                if line.startswith('Test set compression with all inventions applied: '):
                    test_ratio = float(re.match(r'Test set compression with all inventions applied: (.*)', line).group(1))
                elif line.startswith('Time: '):
                    s = float(re.match(r'Time: ([^m]*)ms.*', line).group(1))/1000.
                elif 'Maximum resident set size (kbytes):' in line:
                    mb = float(re.match(r'.*Maximum resident set size \(kbytes\): (.*)', line).group(1))/1000.
            return (train_ratio, test_ratio, s, mb)


    elif mode == 'claim-3':
        with open(path, 'r') as infile:
            steps = [0]
            ratios = [1.0]
            total_steps = None 
            for line in infile:
                if '[new best utility]' in line:
                    step  = re.match(r'.*step=([^ ]*).*', line).group(1)
                    ratio = re.match(r'.*trainratio=([^ ]*).*', line).group(1)
                    steps.append(int(step) + 1)  # they're zero indexed, so add 1
                    ratios.append(float(ratio))
                elif 'Stats {' in line:
                    total_steps = int(re.match(r'.*worklist_steps: ([^,]*),.*', line).group(1))
            return (total_steps, steps, ratios)

    elif mode == 'ablation':

        MAX_TIME = 60 * 90  # max number of seconds of runtime to accept for a run

        # First get the baseline performance
        with open(path + '.stdout', 'r') as basef:
            s = basef.read()
            base_time  = float(re.search(r'User time \(seconds\): ([^\n]*)\n', s).group(1))
            assert base_time <= MAX_TIME and "bytes failed" not in s,  "baseline timed out or ran out of RAM"
            base_steps = int(re.search(r'.*worklist_steps: ([^,]*),.*', s).group(1))

        results = {}
        opts = ("--no-opt-useless-abstract", "--no-opt-force-multiuse", "--no-opt-upper-bound", "--no-opt")
        for opt in opts:
            with open(path + opt + '.stdout', 'r') as f:
                s = f.read()
                time_match = re.search(r'.*User time \(seconds\): ([^\n]*)\n', s)
                if time_match is None or float(time_match.group(1)) > MAX_TIME:
                    results[opt] = "TIME"
                elif "bytes failed" in s:
                    results[opt] = "MEM"
                else:
                    steps = int(re.search(r'.*worklist_steps: ([^,]*),.*', s).group(1))
                    results[opt] = "{:.2f}x".format(float(steps)/float(base_steps))

        return results



mode = sys.argv[1]
path = sys.argv[2]

assert mode in ['claim-2', 'claim-3', 'ablation']


plt.rcParams.update({'font.size': 22})
#workloads = [wl for wl in os.listdir(path) if wl != 'readme.md']
workloads = ['nuts-bolts', 'bridge', 'dials', 'city', 'furniture', 'castle', 'wheels', 'house']
workloads_alt_order = ['nuts-bolts', 'dials', 'furniture',  'wheels',  'bridge', 'city', 'castle', 'house']

wl_to_human_readable = {
    'nuts-bolts': 'nuts & bolts',
    'bridge': 'bridges',
    'dials': 'gadgets',
    'city': 'cities',
    'furniture': 'furniture',
    'castle': 'castles',
    'wheels': 'vehicles',
    'house': 'houses',
}
if mode == 'claim-2':
    num_seeds = int(sys.argv[3])
    for wl in workloads_alt_order:
        seeds = list(range(1, num_seeds))  #os.listdir('/'.join([path, wl]))
        train_ratios = []
        test_ratios = []
        runtimes = []
        mem_usages = []
        for seed in seeds:
            train_ratio, test_ratio, seconds, mb = get_data('/'.join([path, wl, f'{seed}.stderrandout']), mode)
            train_ratios.append(train_ratio)
            test_ratios.append(test_ratio)
            runtimes.append(seconds)
            mem_usages.append(mb)
        train_r_stats = (np.mean(train_ratios), np.std(train_ratios))
        test_r_stats = (np.mean(test_ratios), np.std(test_ratios))
        runtime_stats = (np.mean(runtimes), np.std(runtimes))
        mem_stats = (np.mean(mem_usages), np.std(mem_usages))
        print(f'{wl_to_human_readable[wl]}: \n\ttrain_r_stats={train_r_stats}\n\ttest_r_stats={test_r_stats}\n\truntime_stats={runtime_stats}\n\tmem_stats={mem_stats}')

elif mode == 'claim-3':
    fig = plt.figure(figsize=(10,10))
    markers = ['o', '8', 's', 'p', 'P', '*', 'h', 'D']
    for wl, marker in zip(workloads, markers):
        infile = [f for f in os.listdir('/'.join([path, wl])) if f.endswith('.stderrandout')]
        assert len(infile) == 1
        infile = infile[0]
        total_steps, steps, ratios = get_data('/'.join([path, wl, infile]), 'ex3')

        # make da plot yo
        steps = [100. * s / total_steps for s in steps]
        ratios = [100. * (r - 1.0) / (ratios[-1] - 1.0) for r in ratios]
        ax = plt.step(steps, ratios, marker=marker, where='post', label=wl_to_human_readable[wl], ms=10)
    plt.xscale('log')
    plt.legend(loc='best')
    plt.xlim((0.0001, 100))
    xticks = [0.001, 0.01, 0.1, 1.0, 10, 100]
    plt.xticks(xticks, [f'{t}%' for t in xticks])
    yticks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]#[50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    plt.yticks(yticks, [f'{t}%' for t in yticks])
    plt.grid(True, which='minor', alpha=0.5)
    plt.xlabel('Search progress (log scale; %)')
    plt.ylabel('Reduction in size thus far vs optimal reduction (%)')
    fig.tight_layout()
    plt.savefig(f"cs2-ex3.pdf")

elif mode == 'ablation':

    # Reorder workloads to mimic the table in the paper
    workloads = ['bridge', 'castle', 'city', 'dials', 'furniture', 'house', 'nuts-bolts', 'wheels']
    pt = PrettyTable(['Ablation'] + [wl_to_human_readable[wl] for wl in workloads])
    res_all = {}
    for wl in workloads:
        res = get_data(path + '/stdout/' + wl, mode)
        for opt in res:
            if opt not in res_all:
                res_all[opt] = []
            res_all[opt].append(res[opt])
    pt.add_rows([[k] + res_all[k] for k in sorted(res_all, reverse=True)])
    print(pt)


