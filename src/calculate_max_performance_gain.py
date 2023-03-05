# %%
import pandas as pd
from pathlib import Path
import os
import sys
from itertools import chain


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

def calculate_max_perf_gain(data):
    data_names = [
        'same_m_n_r',
        'along_m', 
        'along_n', 
        'along_r',
    ]
    implementation_names = [
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
    for index_name in data_names:
        subdict = data[index_name]
        main_axis = index_name[-1].lower()
        cycle_lists = []
        for implementation, legend_name, _ in implementation_names:
            if implementation not in subdict:
                print(f'No data found for {index_name}/{implementation}.csv, skipping {legend_name}')
                continue
            df = subdict[implementation]
            # df = df[df['threshold'] == threshold]
            medians = df.groupby(main_axis, as_index=False).median()
            cycle_list = medians['cpu_cycles'].tolist()
            perf_list = (calculate_flops(medians) / medians['cpu_cycles']).tolist()
            cycle_list = [(cycles, legend_name, i*128, performance) for i, (cycles, performance) in enumerate(zip(cycle_list, perf_list), 1)]
            cycle_lists.append(cycle_list)
        cycle_lists_trans = list(zip(*cycle_lists))
        max_runtime_fac, max_val = 0, None
        for row in cycle_lists_trans:
            # m = max(row, key=lambda x: x[0])
            m = row[0] # For baseline
            l = min(row, key=lambda x: x[0])
            runtime_fac = m[0] / l[0]
            if runtime_fac >= max_runtime_fac:
                max_runtime_fac = runtime_fac
                max_val = (m, l)
        max_perf_increase = max_val[0][0] / max_val[1][0]
        print(f'Max. performance increase for {index_name}:\n{max_perf_increase}, at x={max_val[0][2]} ({max_val[1][1]} over {max_val[0][1]})')

        PEAK_PERF = 32.
        max_perf = max(list(chain(*cycle_lists)), key=lambda y: y[3])
        print(f'Peak performance percentage for {index_name}:\n{max_perf[3] / PEAK_PERF}, at x={max_perf[2]} ({max_perf[1]}, with {max_perf[3]} flops/cycle)\n')

def main():
    os.chdir(Path(__file__).parent)
    data_base_path = Path('../output')
    data_folders = list(filter(lambda x: x.is_dir() and x.name != 'VTune', data_base_path.glob('*')))
    index = 0
    if len(data_folders) == 0:
        print('No output folders found. Exiting now.')
        return
    elif len(data_folders) > 1:
        print('Choose which folder to plot:\n')
        for i, folder in enumerate(data_folders):
            print(f'{i}:\t{folder.name}')
        index = int(input('\nFolder number: '))
        if index < 0 or index >= len(data_folders):
            print(f'Illegal argument (not in range): {index}\nExiting now.')
            sys.exit(1)
    data_path = data_folders[index]
    data_subpaths = ['same_m_n_r', 'along_m', 'along_n', 'along_r']
    data_files = [(folder, list((data_path / folder).glob('*.csv'))) for folder in data_subpaths]
    data = {}
    for folder, file_paths in data_files:
        subdata = {file.stem: pd.read_csv(file) for file in file_paths}
        data[folder] = subdata

    calculate_max_perf_gain(data)

if __name__ == '__main__':
    main()
# %%
