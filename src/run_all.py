from distutils.command.build import build
from pathlib import Path
# from stat import S_IXUSR as FILE_EXECUTE_BIT
# from os import popen
import os
import subprocess
from datetime import datetime
import time

build_path = Path('build')
output_path = Path('../output')
PROGRESSBAR_SIZE = 50

def main():
    os.chdir(Path(__file__).parent)
    if (not output_path.exists()):
        output_path.mkdir()
    timestamp = datetime.now().replace(microsecond=0).isoformat().replace(':', '-')
    output_path_timestamp = output_path / timestamp
    output_path_timestamp.mkdir()
    current_progress, number_executables, all_runs = 0, 0, 0

    basic_block_size = 128
    l = [
        ('same_m_n_r', [(i * basic_block_size, i * basic_block_size, i * basic_block_size, (i - 1) * 2) for i in range(1, 11)]),
        ('along_m', [(i * basic_block_size, basic_block_size, basic_block_size, (i - 1) * 2) for i in range(1, 11)]),
        ('along_n', [(basic_block_size, i * basic_block_size, basic_block_size, (i - 1) * 2) for i in range(1, 11)]),
        ('along_r', [(1280, 1280, i * basic_block_size, (i - 1) * 2) for i in range(1, 11)]),
    ]
    number_executables = len([build_path.glob('*.bin')])
    for outname, args in l:
        (output_path_timestamp / outname).mkdir()
        all_runs += number_executables * len(args)
     
    for f in build_path.glob('*.bin'):
        # if f.stat().st_mode & FILE_EXECUTE_BIT:
        for outname, args in l:
            print(f'Executing file {f.stem} in mode \"{outname.replace("_", " ")}\"')
            outputfile = output_path_timestamp / outname / (f.stem + '.csv')
            outputfile_text = str(outputfile)
            for m, n, r, v in args:
                cmd = f'{str(f)} {outputfile_text} {m} {n} {r} {v}'
                result = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f'Command: {cmd}, with PID: {result.pid}')
                progress = int(PROGRESSBAR_SIZE * current_progress / all_runs)
                print(f'Progress: [{"+" * progress}{"." * (PROGRESSBAR_SIZE - progress)}]')
                while result.poll() is None:
                    if (stdout := result.stdout.readline().strip().decode()) != '':
                        print(stdout)
                    time.sleep(0.1)
                while (stdout := result.stdout.readline().strip().decode()) != '':
                    print(stdout)
                if (error_code := result.poll()) != 0:
                    print(f'Error code was not zero: {error_code=}')
                    if input('Continue? [y/n] ').strip().lower()[0] == 'n':
                        quit()
                print('\n')
                current_progress += 1





if __name__ == '__main__':
    main()
