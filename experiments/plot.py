import sys
import os
import re

import matplotlib.pyplot as plt 
import numpy as np

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
    with open(path, 'r') as infile:
        if mode == 'claim-2':
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


        elif mode == 'ex2':
            target_line = None
            for line in infile:
                if line.startswith('Cost Improvement (test): '):
                    target_line = line
                    break
            if target_line is None:
                print(path)
                return None

            ratio = re.match(r'Cost Improvement \(test\): \(([^x]*)x.*', target_line).group(1)

            return float(ratio)

        elif mode == 'ex3':
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



mode = sys.argv[1]
path = sys.argv[2]

assert mode in ['claim-2', 'ex2', 'ex3']


plt.rcParams.update({'font.size': 22})
#workloads = [wl for wl in os.listdir(path) if wl != 'readme.md']
workloads = ['nuts-bolts', 'bridge', 'dials', 'city', 'furniture', 'castle', 'wheels', 'house']
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
    for wl in workloads:
        seeds = list(range(1, 51))  #os.listdir('/'.join([path, wl]))
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

elif mode == 'ex2':
    color = plt.cm.get_cmap('viridis')(0.)
    fig, hosts = plt.subplots(4, 2, figsize=(18,15), sharex=True)
    for idx, (wl, host) in enumerate(zip(workloads, hosts.flatten())):
        ratios_means   = []
        ratios_stds    = []
        splits         = set()
        for split in os.listdir('/'.join([path, wl])):
            splits.add(int(split))

        splits = list(splits)
        splits.sort()
        for split in splits:
            split = str(split)

            ra_local = [] 

            seeds = os.listdir('/'.join([path, wl, split]))
            for f in seeds:
                if f.endswith('.stderrandout'):
                    ratio = get_data('/'.join([path, wl, split, f]), 'ex2')
                    if ratio is None:
                        continue

                    ra_local.append(ratio)

            ratios_means.append(np.mean(ra_local))
            ratios_stds.append(np.std(ra_local))

        # make da plot yo
            
        #if idx > 5:
        #    host.set_xlabel("Training set split %")
        #if idx % 2 == 0:
        #    host.set_ylabel("Test set compression ratio")
        
        p1, = host.plot(splits, ratios_means, color=color, marker='o')
        host.fill_between(splits, np.subtract(ratios_means, ratios_stds), np.add(ratios_means, ratios_stds), color=color, alpha=0.2)
        
        host.set_title(wl_to_human_readable[wl])

        #host.legend(loc='best')

    fig.supxlabel('Training set sample size (%)')
    fig.supylabel('Test set compression ratio')
    #fig.text(0.5, 0.04, 'Training set sample size (%)', ha='center')
    #fig.text(0.04, 0.5, 'Test set compression ratio', va='center', rotation='vertical')
    fig.tight_layout()
    plt.savefig(f"cs2-ex2.pdf")

elif mode == 'ex3':
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
    #fig.set_title(wl_to_human_readable[wl])
    plt.legend(loc='best')
    #host.set_xlim((0.5, total_steps))
    plt.xlim((0.0001, 100))
    xticks = [0.001, 0.01, 0.1, 1.0, 10, 100]
    plt.xticks(xticks, [f'{t}%' for t in xticks])
    yticks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]#[50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    plt.yticks(yticks, [f'{t}%' for t in yticks])
    plt.grid(True, which='minor', alpha=0.5)
    plt.xlabel('Search progress (log scale; %)')
    plt.ylabel('Reduction in size thus far vs optimal reduction (%)')
    #host.set_xlim((-1, 100))
    #fig, hosts = plt.subplots(4, 2, figsize=(18,15), sharex=True, sharey=True)
    #for idx, (wl, host) in enumerate(zip(workloads, hosts.flatten())):
    #    infile = [f for f in os.listdir('/'.join([path, wl])) if f.endswith('.stderrandout')]
    #    assert len(infile) == 1
    #    infile = infile[0]
    #    total_steps, steps, ratios = get_data('/'.join([path, wl, infile]), 'ex3')

    #    # make da plot yo
    #    steps = [100. * s / total_steps for s in steps]
    #    ratios = [100. * r / ratios[-1] for r in ratios]
    #    p1 = host.step(steps, ratios, color=color, marker='o', where='post')
    #    print(steps)
    #    print(ratios)
    #    host.set_xscale('log')
    #    host.set_title(wl_to_human_readable[wl])
    #    #host.set_xlim((0.5, total_steps))
    #    host.set_xlim((0.5, 100))
    #    #host.set_xlim((-1, 100))

    #fig.supxlabel('Number of nodes explored')
    #fig.supylabel('Training set compression ratio')
    fig.tight_layout()
    plt.savefig(f"cs2-ex3.pdf")

    #fig, host = plt.subplots(figsize=(8,5))
    #    
    #par1 = host.twinx()
    #par2 = host.twinx()
    #    
    ##host.set_xlim(0, 2)
    ##host.set_ylim(0, 2)
    ##par1.set_ylim(0, 4)
    ##par2.set_ylim(1, 65)
    #    
    #host.set_xlabel("Training set split %")
    #host.set_ylabel("Test set compression ratio")
    #par1.set_ylabel("Invention size")
    #par2.set_ylabel("Number of usages of invention")
    
    #color1 = plt.cm.get_cmap('viridis')(0.)
    #color2 = plt.cm.get_cmap('viridis')(0.5)
    #color3 = plt.cm.get_cmap('viridis')(0.9)
    
    #p1, = host.plot(splits, ratios_means, color=color1, label="Test set compression ratio"   , marker='o')
    #host.fill_between(splits, np.subtract(ratios_means, ratios_stds), np.add(ratios_means, ratios_stds), color=color1, alpha=0.2)
    #p2, = par1.plot(splits, sizes_means,  color=color2, label="Invention size"               , marker='o')
    #par1.fill_between(splits, np.subtract(sizes_means, sizes_stds), np.add(sizes_means, sizes_stds), color=color2, alpha=0.2)
    #p3, = par2.plot(splits, usages_means, color=color3, label="Number of usages of invention", marker='o')
    #par2.fill_between(splits, np.subtract(usages_means, usages_stds), np.add(usages_means, usages_stds), color=color3, alpha=0.2)
    
    #lns = [p1, p2, p3]
    #host.legend(handles=lns, loc='best')
    
    ## right, left, top, bottom
    #par2.spines['right'].set_position(('outward', 60))
    
    ## no x-ticks                 
    ##par2.xaxis.set_ticks([])
    
    ## Sometimes handy, same for xaxis
    ##par2.yaxis.set_ticks_position('right')
    
    ## Move "Velocity"-axis to the left
    ## par2.spines['left'].set_position(('outward', 60))
    ## par2.spines['left'].set_visible(True)
    ## par2.yaxis.set_label_position('left')
    ## par2.yaxis.set_ticks_position('left')
    
    #host.yaxis.label.set_color(p1.get_color())
    #par1.yaxis.label.set_color(p2.get_color())
    #par2.yaxis.label.set_color(p3.get_color())
    
    ## Adjust spacings w.r.t. figsize
    #fig.tight_layout()
    ## Alternatively: bbox_inches='tight' within the plt.savefig function 
    ##                (overwrites figsize)
    
    ## Best for professional typesetting, e.g. LaTeX
    #plt.savefig(f"cs2-{wl}.pdf")
    ## For raster graphics use the dpi argument. E.g. '[...].png", dpi=200)'
