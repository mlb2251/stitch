import sys
import os
import matplotlib.pyplot as plt 
import numpy as np

plt.style.use('ggplot')

path = sys.argv[1]

for wl in os.listdir(path):
    # one plot per workload!
    runtimes_means = []
    ratios_means   = []
    sizes_means    = []
    usages_means   = []
    runtimes_stds  = []
    ratios_stds    = []
    sizes_stds     = []
    usages_stds    = []
    splits         = []
    for split in os.listdir('/'.join([path, wl])):
        splits.append(int(split))
        rt_local = numpy.array([])
        ra_local = numpy.array([])
        s_local  = numpy.array([])
        u_local  = numpy.array([])

        for f in os.listdir('/'.join([path, wl, split])):
            if f.endswith('.stderrandout'):
                print((wl, split, f))
                runtime, ratio, size, usage = get_data('/'.join([path, wl, split, f]))
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


    plt.plot(groups, single_mean, label = 'One goal', marker = 'o')
    plt.fill_between(groups, single_mean - single_std, single_mean + single_std, alpha = 0.2)
    plt.plot(groups, multi_mean, label = 'Two goals', marker = 'o')
    plt.fill_between(groups, multi_mean - multi_std, multi_mean + multi_std, alpha = 0.2)
    plt.axhline(1.0, linestyle='--', label = 'Expert')

    plt.margins(x=0)
    plt.xlabel("Approx. Minutes of Training Data")
    plt.ylabel("Success Rate (25 runs)")
    plt.title("HelloAgentic performance with one or two goals")
    plt.legend()





    fig, host = plt.subplots(figsize=(8,5))
        
    par1 = host.twinx()
    par2 = host.twinx()
        
    host.set_xlim(0, 2)
    host.set_ylim(0, 2)
    par1.set_ylim(0, 4)
    par2.set_ylim(1, 65)
        
    host.set_xlabel("Training set split %")
    host.set_ylabel("Test set compression ratio")
    par1.set_ylabel("Invention size")
    par2.set_ylabel("Number of usages of invention")
    
    #color1 = plt.cm.viridis(0)
    #color2 = plt.cm.viridis(0.5)
    #color3 = plt.cm.viridis(.9)
    
    p1, = host.plot(splits, ratios_means,  label="Test set compression ratio")
    p2, = par1.plot([0, 1, 2], sizes_means,    label="Invention size")
    p3, = par2.plot([0, 1, 2], usages_means, label="Number of usages of invention")
    
    lns = [p1, p2, p3]
    host.legend(handles=lns, loc='best')
    
    # right, left, top, bottom
    par2.spines['right'].set_position(('outward', 60))
    
    # no x-ticks                 
    par2.xaxis.set_ticks([])
    
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

def get_data(path):
    return (1, 1, 1, 1)
