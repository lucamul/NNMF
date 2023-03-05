# %%
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import sys

def calculate_flops(df):
    Q_RATIO = 0.25
    m = df['m']
    n = df['n']
    r = df['r']
    num_iter = df['loop iterations']
    q = n * Q_RATIO

    # First count Luca
    # mults = num_iter * (5*m*n*r + r*r*n + r*n + m*r) + m*n*r
    # adds = num_iter * (5*m*n*r + r*r*n - m*n - 2*r*n - r*r - 2*m*r) + m*n*r
    # divs = num_iter * (m*r + r*n)
    
    # Current flop count computation
    adds = num_iter * (5*m*n*r + m*n + r*m*r + r*r*n) + m*n*r
    mults = num_iter * (5*m*n*r + r*m*r + r*r*n + r*n) + m*n*r
    divs = num_iter * (m*r + r*n)

    # Old flop count init
    # flops_init = m*n*r*q + m*n + m*r + n    
    
    # Current flop count init
    # flops_init = 2*m*n + n + r*q*m + r*m    
    flops_init = 0

    return mults + adds + divs + flops_init

def make_perf_plot(data, plot_path, limit_plots=None, separate_plots=False):
    plt.style.use(['seaborn', 'seaborn-poster', 'seaborn-ticks'])
    if separate_plots:
        plts = [
            ('same_m_n_r', 'M = N = R'), 
            ('along_m', 'M, with N = R = 128'), 
            ('along_n', 'N, with M = R = 128'), 
            ('along_r', 'R, with M = N = 1280')
        ]
        plts = [(a, b, plot_path.parent / f'{plot_path.stem}_{i}{plot_path.suffix}') for i, (a, b) in enumerate(plts, 1)]
        plot_order = [
            ('basic', 'Basic', 'b'),
            ('blas', 'BLAS', 'c'),
            ('loop_reordering', 'Loop Gathering', 'darkorange'),
            ('ilp', 'ILP', 'g'),
            ('blocking_caches', 'Blocking for Cache', 'r'),
            ('blocking_registers', 'Blocking for Registers', 'purple'),
            #('blocking_ilp', 'Block for Registers - ILP', 'brown'),
            ('blocking_vectorize_128', 'Vectorization - 128', 'magenta'),
            ('blocking_vectorize_256', 'Vectorization - 256', 'gray'),
        ]
        if limit_plots is not None:
            plot_order = plot_order[:limit_plots]
        
        for index_name, subtitle, plot_name in plts:
            fig, ax = plt.subplots()
            plt.grid(True, color='w', linestyle='-', linewidth=1, axis='y')
            plt.gca().patch.set_facecolor('0.9')
            plt.title('Performance Plot - Intel i7-7500U (Skylake)', loc='center', weight='bold', y=1.08)
            plt.suptitle(f"Measured along {subtitle}\nGCC 9.4.0 with flags -O3 -ffast-math -march=native", weight="bold", fontsize=14, y=0.94)
            # plt.ticklabel_format(axis='y', style='plain', useMathText=True, scilimits=(0,0))
            ax.set_ylabel("Performance \n[flops/cycle]", rotation='horizontal', size=14)
            ax.yaxis.set_label_coords(0.0, 1.03)
            ax.set_xlabel(f"Input size {subtitle.split(',')[0]}", size=14)
            ax.set_xticks([i * 128 for i in range(12)])
            ax.set_xticklabels(str(i * 128) if i > 0 and i % 2 == 0 else '' for i in range(12))
            
            subdict = data[index_name]
            main_axis = subtitle[0].lower()
            for implementation, legend_name, color in plot_order:
                if implementation not in subdict:
                    print(f'No data found for {index_name}/{implementation}.csv, skipping {legend_name}')
                    continue
                df = subdict[implementation]
                # df = df[df['threshold'] == threshold]
                medians = df.groupby(main_axis, as_index=False).median()
                ax.plot(medians[main_axis], calculate_flops(medians) / medians['cpu_cycles'], linestyle='--', marker='.', color=color, label=legend_name)
                # subplot.plot(medians[main_axis], medians['flops'] / medians['cpu_cycles'], linestyle='--', marker='.', label=path.split('/')[-1][:-4])
            ax.legend(prop={'size': 11})
            plt.savefig(plot_name, bbox_inches='tight', dpi=400)
            # plt.show()
    else:
        fig, ax = plt.subplots(2, 2)
        fig.suptitle(f"Performance Plot - Intel i7-7500U (Skylake)\nMeasured with flags -O3 -ffast-math -march=native and GCC version 9.4.0", weight="bold", fontsize=18)
        plt_same, plt_m, plt_n, plt_r = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

        plts = [
            ('same_m_n_r', 'M = N = R', plt_same), 
            ('along_m', 'M, with N = R = 128', plt_m), 
            ('along_n', 'N, with M = R = 128', plt_n), 
            ('along_r', 'R, with M = N = 1280', plt_r)
        ]
        plot_order = [
            ('basic', 'Basic', 'b'),
            ('blas', 'BLAS', 'c'),
            ('loop_reordering', 'Loop Gathering', 'darkorange'),
            ('ilp', 'ILP', 'g'),
            ('blocking_caches', 'Blocking for Cache', 'r'),
            ('blocking_registers', 'Blocking for Registers', 'purple'),
            #('blocking_ilp', 'Block for Registers - ILP', 'brown'),
            ('blocking_vectorize_128', 'Vectorization - 128', 'magenta'),
            ('blocking_vectorize_256', 'Vectorization - 256', 'gray'),
        ]
        if limit_plots is not None:
            plot_order = plot_order[:limit_plots]

        for index_name, subtitle, subplot in plts:
            subplot.set_ylabel("Performance \n[flops/cycle]", rotation='horizontal', size=14)
            subplot.yaxis.set_label_coords(0.0, 1.03)
            subplot.set_xlabel(f"Input size {subtitle.split(',')[0]}", size=14)
            subplot.set_title(f'Along {subtitle}')
            subplot.set_facecolor('0.9')
            subplot.grid(True, color='w', linestyle='-', linewidth=1, axis='y')
            subplot.set_xticks([i * 128 for i in range(12)])
            subplot.set_xticklabels(str(i * 128) if i > 0 and i % 2 == 0 else '' for i in range(12))
            
            subdict = data[index_name]
            main_axis = subtitle[0].lower()
            for implementation, legend_name, color in plot_order:
                if implementation not in subdict:
                    print(f'No data found for {index_name}/{implementation}.csv, skipping {legend_name}')
                    continue
                df = subdict[implementation]
                # df = df[df['threshold'] == threshold]
                medians = df.groupby(main_axis, as_index=False).median()
                subplot.plot(medians[main_axis], calculate_flops(medians) / medians['cpu_cycles'], linestyle='--', marker='.', color=color, label=legend_name)
                # subplot.plot(medians[main_axis], medians['flops'] / medians['cpu_cycles'], linestyle='--', marker='.', label=path.split('/')[-1][:-4])
            # subplot.legend()
        fig.tight_layout(pad=2, h_pad=2, w_pad=3)
        plt_m.legend(loc='upper right', bbox_to_anchor=(1.7, 0.9))
        plt.savefig(plot_path, bbox_inches='tight', dpi=400)
        # plt.show()

def main():
    os.chdir(Path(__file__).parent)
    plot_path = Path('../plots')
    data_base_path = Path('../output')
    if not plot_path.exists():
        plot_path.mkdir()
    plot_path.is_dir
    plot_folders = list(filter(lambda x: x.is_dir() and x.name != 'VTune', data_base_path.glob('*')))
    index = 0
    if len(plot_folders) == 0:
        print('No output folders found. Exiting now.')
        return
    elif len(plot_folders) > 1:
        print('Choose which folder to plot:\n')
        for i, folder in enumerate(plot_folders):
            print(f'{i}:\t{folder.name}')
        index = int(input('\nFolder number: '))
        if index < 0 or index >= len(plot_folders):
            print(f'Illegal argument (not in range): {index}\nExiting now.')
            sys.exit(1)
    data_path = plot_folders[index]
    data_subpaths = ['same_m_n_r', 'along_m', 'along_n', 'along_r']
    # data_subpaths_all = ['same_m_n_r', 'along_m', 'along_n', 'along_r']
    # data_subpaths = list(filter(lambda folder: (data_path / folder).exists(), data_subpaths_all))
    data_files = [(folder, list((data_path / folder).glob('*.csv'))) for folder in data_subpaths]
    data = {}
    for folder, file_paths in data_files:
        subdata = {file.stem: pd.read_csv(file) for file in file_paths}
        data[folder] = subdata

    # plots = [['output/snd_meeting/' + prefix + '/' + suffix for suffix in 'basic,blas,blocking,inlined'.split('')] for prefix in 'along_m,along_n,along_r,same_m_n_r'.split(',')]
    make_perf_plot(data, plot_path / f'perfplot_{data_path.name}.png')
    # make_perf_plot(data, plot_path / f'perfplot_{data_path.name}.png', separate_plots=True)
    make_perf_plot(data, plot_path / f'perfplot_{data_path.name}.eps', separate_plots=True)
    # for i in range(10):
    #     make_perf_plot(data, plot_path / f'plot{str(i)}.png', limit_plots=i+1)

if __name__ == '__main__':
    main()
# %%
