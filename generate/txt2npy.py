import os
import sys
import argparse
import math
import numpy as np


def main(args):
    in_dir = args.input_folder
    out_dir = args.output_folder
    resolution = args.resolution

    file_names = ['chr{}_{}Kb.txt'.format(i, resolution) for i in range(5, 23)]

    for file_name in file_names:
        file = open(os.path.join(in_dir, file_name), 'r')
        _list = file.readlines()

        min_num = max_num = int(int(_list[0].split()[0]) / 1000)
        for i in range(len(_list)):
            line = _list[i].split()
            x = int(int(line[0]) / 1000)
            y = int(int(line[1]) / 1000)
            min_num = min(min_num, x, y)
            max_num = max(max_num, x, y)

        matrix = np.zeros((max_num+1, max_num+1), dtype=np.uint16)

        for i in range(len(_list)):
            line = _list[i].split()
            x = int(int(line[0]) / 1000)
            y = int(int(line[1]) / 1000)
            z = float(line[2])
            z = 0 if math.isnan(z) else min(int(z), 65536)
            matrix[x, y] = matrix[y, x] = z

        prefix, ext = os.path.splitext(file_name)
        np.savez_compressed(os.path.join(out_dir, prefix), hic=matrix)
        # np.savez_compressed(os.path.join(path, 'chr6_1Kb_cut'), hic=matrix[0:400, 0:400])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate train data')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-i', dest='input_folder', help='', required=True)
    req_args.add_argument('-o', dest='output_folder', help='', required=True)

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-r', dest='resolution',
                           help='resolution(kb)[default:1]',
                           default=1)

    args = parser.parse_args(sys.argv[1:])
    main(args)