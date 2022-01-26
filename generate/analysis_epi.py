# bigwig -> npy
import os
import sys
import argparse
import numpy as np
import pyBigWig


def mkdir(out_dir):
    if not os.path.isdir(out_dir):
        print(f'Making directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)


def main(args):
    input_file = args.input_file
    output_folder = args.output_folder
    step_length = args.resolution
    mkdir(output_folder)

    bw = pyBigWig.open(input_file)
    print(bw.isBigWig())
    print(bw.chroms())
    for chrom in bw.chroms():
        print(chrom)
        peaks = np.zeros(int(bw.chroms(chrom)/step_length))
        write_start, length, sum_pos = 0, 0, 0
        entries = bw.intervals(chrom)
        if entries is None:
            continue
        for entry in entries:
            start, end, pos = entry
            old_length = length
            length += end - start
            if length < step_length:
                sum_pos += (end - start) * pos
                continue
            while length >= step_length:
                sum_pos += (step_length - old_length) * pos
                old_length = 0
                peaks[int(write_start/step_length)] = sum_pos / step_length
                length -= step_length
                write_start += step_length
                sum_pos = 0
            sum_pos += length * pos
        np.save(os.path.join(output_folder, '{}_{}b'.format(chrom, step_length)), peaks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis epi data')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-i', dest='input_file', help='', required=True)
    req_args.add_argument('-o', dest='output_folder', help='', required=True)

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-r', dest='resolution', type=int,
                           help='resolution(b)[default:1000]',
                           default=1000)
    
    args = parser.parse_args(sys.argv[1:])
    main(args)