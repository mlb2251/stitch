import sys
import os
import re

import matplotlib.pyplot as plt 
import numpy as np

plt.style.use('ggplot')


def get_data(path):
    #print(path)
    #input()
    with open(path, 'r') as infile:
        target_line = None
        for line in infile:
            if line.startswith('train: '):
                target_line = line
                break
        if target_line is None:
            return None

        ratio = re.match(r'.*multiplier \(test\): ([^x]*)x.*', target_line).group(1)
        size = re.match(r'.*size: ([^ ]*) .*', target_line).group(1)
        usages = re.match(r'.*uses \(train\): ([^ ]*) .*', target_line).group(1)
        
        return (1., float(ratio), int(size), int(usages))


path = sys.argv[1]

workloads = os.listdir(path)
for wl in workloads:
    if wl == 'readme.md': continue

    # one plot per workload!
    runtimes_means = []
    ratios_means   = []
    sizes_means    = []
    usages_means   = []
    runtimes_stds  = []
    ratios_stds    = []
    sizes_stds     = []
    usages_stds    = []
    splits         = set()
    for split in os.listdir('/'.join([path, wl])):
        splits.add(int(split))

    splits = list(splits)
    splits.sort()
    #print(splits)
    #input()
    for split in splits:
        split = str(split)

        rt_local = [] 
        ra_local = [] 
        s_local  = [] 
        u_local  = [] 

        files = os.listdir('/'.join([path, wl, split]))
        #print(files)
        #input()
        for f in files:
            if f.endswith('.stderrandout'):
                #print(f)
                #input()
                data = get_data('/'.join([path, wl, split, f]))
                if data is None:
                    continue
                runtime, ratio, size, usage = data
                rt_local.append(runtime)
                ra_local.append(ratio)
                s_local.append(size)
                u_local.append(usage)

        runtimes_means.append(np.mean(rt_local))
        runtimes_stds.append(np.std(rt_local))
        ratios_means.append(np.mean(ra_local))
        ratios_stds.append(np.std(ra_local))
        sizes_means.append(np.mean(s_local))
        sizes_stds.append(np.std(s_local))
        usages_means.append(np.mean(u_local))
        usages_stds.append(np.std(u_local))

        #splits.append(int(split))


    #plt.plot(groups, single_mean, label = 'One goal', marker = 'o')
    #plt.fill_between(groups, single_mean - single_std, single_mean + single_std, alpha = 0.2)
    #plt.plot(groups, multi_mean, label = 'Two goals', marker = 'o')
    #plt.fill_between(groups, multi_mean - multi_std, multi_mean + multi_std, alpha = 0.2)
    #plt.axhline(1.0, linestyle='--', label = 'Expert')

    #plt.margins(x=0)
    #plt.xlabel("Approx. Minutes of Training Data")
    #plt.ylabel("Success Rate (25 runs)")
    #plt.title("HelloAgentic performance with one or two goals")
    #plt.legend()





    fig, host = plt.subplots(figsize=(8,5))
        
    par1 = host.twinx()
    par2 = host.twinx()
        
    #host.set_xlim(0, 2)
    #host.set_ylim(0, 2)
    #par1.set_ylim(0, 4)
    #par2.set_ylim(1, 65)
        
    host.set_xlabel("Training set split %")
    host.set_ylabel("Test set compression ratio")
    par1.set_ylabel("Invention size")
    par2.set_ylabel("Number of usages of invention")
    
    color1 = plt.cm.get_cmap('viridis')(0.)
    color2 = plt.cm.get_cmap('viridis')(0.5)
    color3 = plt.cm.get_cmap('viridis')(0.9)
    
    p1, = host.plot(splits, ratios_means, color=color1, label="Test set compression ratio"   , marker='o')
    host.fill_between(splits, np.subtract(ratios_means, ratios_stds), np.add(ratios_means, ratios_stds), color=color1, alpha=0.2)
    p2, = par1.plot(splits, sizes_means,  color=color2, label="Invention size"               , marker='o')
    par1.fill_between(splits, np.subtract(sizes_means, sizes_stds), np.add(sizes_means, sizes_stds), color=color2, alpha=0.2)
    p3, = par2.plot(splits, usages_means, color=color3, label="Number of usages of invention", marker='o')
    par2.fill_between(splits, np.subtract(usages_means, usages_stds), np.add(usages_means, usages_stds), color=color3, alpha=0.2)
    
    lns = [p1, p2, p3]
    host.legend(handles=lns, loc='best')
    
    # right, left, top, bottom
    par2.spines['right'].set_position(('outward', 60))
    
    # no x-ticks                 
    #par2.xaxis.set_ticks([])
    
    # Sometimes handy, same for xaxis
    #par2.yaxis.set_ticks_position('right')
    
    # Move "Velocity"-axis to the left
    # par2.spines['left'].set_position(('outward', 60))
    # par2.spines['left'].set_visible(True)
    # par2.yaxis.set_label_position('left')
    # par2.yaxis.set_ticks_position('left')
    
    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    
    # Adjust spacings w.r.t. figsize
    fig.tight_layout()
    # Alternatively: bbox_inches='tight' within the plt.savefig function 
    #                (overwrites figsize)
    
    # Best for professional typesetting, e.g. LaTeX
    plt.savefig(f"cs2-{wl}.pdf")
    # For raster graphics use the dpi argument. E.g. '[...].png", dpi=200)'

