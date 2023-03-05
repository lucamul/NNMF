from distutils.command.build import build
from pathlib import Path
from stat import S_IXUSR as FILE_EXECUTE_BIT
# from os import popen
import os
from subprocess import run

output_path = 'output/block_sizes_measurements.csv'

def main():
    os.chdir(Path(__file__).parent)
    cache_block_sizes = [int(2**i) for i in range(3, 9)]
    register_block_sizes = [int(2**i) for i in range(1, 6)]
    for cache_block_size in cache_block_sizes:
        for register_block_size in filter(lambda x: x < cache_block_size, register_block_sizes):
            compile_args = f'gcc -O3 -DBLOCKSIZE={cache_block_size} -DNU={register_block_size} -DNUM_ITERATIONS=1 -DNUM_V=1 main.c -o build/test_blocksize init_w.h init_w.c utils.h utils.c nnmf_common.h nnmf/opt_4.c nnmf.h nnmf.c -lm'
            run_args = f'./build/test_blocksize {output_path} 128 128 128 {cache_block_size * 1000 + register_block_size} 3'
            print(f'Compile for {cache_block_size=} and {register_block_size=}')
            run(compile_args.split(' '))
            print(f'Run for {cache_block_size=} and {register_block_size=}')
            run(run_args.split(' '))

if __name__ == '__main__':
    main()