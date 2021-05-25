import os
import sys
import argparse
from math import ceil
import numpy as np
import numba as nb

@nb.njit()
def point_add(peaks):
    length = peaks.shape[0]
    epi_matrix = np.zeros((length, length), dtype=np.uint16)
    for i in range(length):
        for j in range(length):
            epi_matrix[i, j] = peaks[i] + peaks[j]
    return epi_matrix


def divide(matrix, focus_size, subimage_size):
    rows, cols = matrix.shape
    sub_rows, sub_cols = ceil(rows / subimage_size), round(focus_size / subimage_size)
    new_matrix = np.zeros((sub_rows, sub_cols, subimage_size, subimage_size), dtype=np.uint16)
    for m in range(sub_rows):
        i = m * subimage_size
        offset = m
        for n in range(sub_cols):
            j = (offset + n) * subimage_size
            if j >= cols:
                break
            tmp_matrix = matrix[i:i+subimage_size, j:j+subimage_size]
            # 补零
            _y = i + subimage_size - rows if i + subimage_size > rows else 0
            _x = j + subimage_size - cols if j + subimage_size > cols else 0
            tmp_matrix = np.pad(tmp_matrix, ((0,_y),(0,_x)), 'constant', constant_values=(0, 0))

            new_matrix[m, n] = tmp_matrix
    return new_matrix


def main(args):
    in_dir = args.input_folder
    out_dir = args.output_folder
    resolution = args.resolution
    subimage_size = args.subimage_size
    focus_size = args.focus_size
    if focus_size % subimage_size != 0:
        raise Exception()

    # file_names = ['chr{}_{}kb.npy'.format(i, resolution) for i in list(range(5, 23)) + ['X', 'Y']]
    file_names = ['chr{}_{}b.npz'.format(i, resolution) for i in range(6, 23)]

    for file_name in file_names:
        file_path = os.path.join(in_dir, file_name)
        print(file_path)

        epi_matrix = np.load(file_path)['epi']
        # Min, Max = np.min(peaks), np.max(peaks)
        # peaks = (peaks - Min) / (Max - Min)
        # peaks = 1 - peaks
        # peaks = (peaks * 10000).astype(np.uint16)

        # epi_matrix = point_add(peaks)
        epi_matrix = divide(epi_matrix, focus_size, subimage_size)
        
        prefix, ext = os.path.splitext(file_name)
        np.savez_compressed(os.path.join(out_dir, prefix), epi=epi_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate train data')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-i', dest='input_folder', help='', required=True)
    req_args.add_argument('-o', dest='output_folder', help='', required=True)

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-r', dest='resolution',
                           help='resolution(kb)[default:1000]',
                           default=1000)
    misc_args.add_argument('-s', dest='subimage_size',
                           help='The size of the captured image[default:400]',
                           default=400)
    misc_args.add_argument('-f', dest='focus_size',
                           help='The size of the picture to follow[default:400]',
                           default=2000)

    args = parser.parse_args(sys.argv[1:])
    main(args)