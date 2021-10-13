import os
import sys
from math import ceil
import argparse
import numpy as np
import numba as nb

max_limit = 10000
nan_num = 32767


def mkdir(out_dir):
    if not os.path.isdir(out_dir):
        print(f'Making directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)


@nb.jit()
def make_matrix(matrix, features, focus_size):
    length = features.shape[0]
    for _i in range(length):
        for _j in range(_i, min(_i+focus_size, length)):
            _num = max_limit * np.mean(np.append(features[_i, :], features[_j, :]))
            if features[_i, :].max() == 0 or features[_j, :].max() == 0:
                matrix[_i, _j] = matrix[_j, _i] = _num
            else:
                _pearson = np.corrcoef(features[_i, :], features[_j, :])
                matrix[_i, _j] = nan_num if np.isnan(_pearson[0, 1]) else int(_pearson[0, 1] * _num)
                matrix[_j, _i] = nan_num if np.isnan(_pearson[1, 0]) else int(_pearson[0, 1] * _num)
            
            if matrix[_i, _j] > max_limit:
                print(features[_i, :])
                print(features[_j, :])
    return matrix


def divide(matrix, focus_size, subimage_size):
    rows, cols = matrix.shape
    sub_rows, sub_cols = ceil(rows / subimage_size), round(focus_size / subimage_size)
    new_matrix = np.zeros((sub_rows, sub_cols, subimage_size, subimage_size), dtype=np.int32)
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
            tmp_matrix = np.pad(tmp_matrix, ((0,_y),(0,_x)), 'constant', constant_values=(max_limit, max_limit))

            new_matrix[m, n] = tmp_matrix
    return new_matrix


def main(args):
    in_dir = args.input_folder
    out_dir = args.output_folder
    resolution = args.resolution
    subimage_size = args.subimage_size
    focus_size = args.focus_size
    print(args)
    if focus_size % subimage_size != 0:
        raise Exception()
    mkdir(out_dir)

    all_path = []
    for (_, dirnames, _) in os.walk(in_dir):
        for dirname in dirnames:
            all_path.append(os.path.join(in_dir, dirname))
        break

    # file_names = ['chr{}_{}b.npy'.format(i, resolution) for i in list(range(5, 23)) + ['X', 'Y']]
    file_names = ['chr{}_{}b.npy'.format(i, resolution) for i in range(1, 23)]

    for file_name in file_names:
        print(file_name)
        # 构造特征矩阵
        epi_info = []
        for dir_path in all_path:
            file_path = os.path.join(dir_path, file_name)
            _epi = np.load(file_path)

            Max, Min = _epi.max(), _epi.min()
            _epi = (_epi - Min) / (Max - Min)

            epi_info.append(_epi)
        epi_info = np.array(epi_info)
        epi_info = epi_info.T
        print(epi_info.shape)

        epi_matrix = np.zeros((epi_info.shape[0], epi_info.shape[0]), dtype=np.int32)
        epi_matrix = make_matrix(epi_matrix, epi_info, focus_size)
        epi_matrix = divide(epi_matrix, focus_size, subimage_size)

        prefix, ext = os.path.splitext(file_name)
        np.savez_compressed(os.path.join(out_dir, prefix), epi=epi_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate epi train data')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-i', dest='input_folder', help='', required=True)
    req_args.add_argument('-o', dest='output_folder', help='', required=True)

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-r', dest='resolution', type=int,
                           help='resolution(b)[default:1000]',
                           default=1000)
    misc_args.add_argument('-s', dest='subimage_size', type=int,
                           help='The size of the captured image[default:400]',
                           default=400)
    misc_args.add_argument('-f', dest='focus_size', type=int,
                           help='The size of the picture to follow[default:400]',
                           default=2000)
    
    args = parser.parse_args(sys.argv[1:])
    main(args)