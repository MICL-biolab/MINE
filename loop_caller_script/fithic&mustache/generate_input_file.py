import os
import sys
import gzip
import argparse
import numpy as np
import numba as nb
from loop_caller_script.tools import merge_matrix, mkdir, get_Several_MinMax_Array

@nb.jit
def calculate_fragments(chrom, resolution, matrix, fragments):
    dim = matrix.shape
    for i in range(dim[0]):
        fragments[i] = np.array([chrom, 0, i*resolution, matrix[i, i:i+2000].sum()*2, 0])
    return fragments

def main(args):
    input_folder = args.input_folder
    out_dir = args.output_folder
    resolution = args.resolution
    is_predict = args.is_predict

    mkdir(out_dir)
    if is_predict is True:
        out_dir = os.path.join(out_dir, 'enhanced')
    else:
        out_dir = os.path.join(out_dir, 'hr')
    mkdir(out_dir)

    for chrom in range(1, 23):
        matrix_path = os.path.join(input_folder, 'chr{}_{}b.npz'.format(chrom, resolution))
        matrix = np.load(matrix_path)
        _out_dir = os.path.join(out_dir, 'chr{}_{}b'.format(chrom, resolution))
        mkdir(_out_dir)
        print(matrix_path)

        if is_predict is True:
            matrix = matrix['out']
            for i in range(matrix.shape[0]):
                matrix[i][0] = (matrix[i][0] + matrix[i][0].T)/2
            matrix[matrix>255]=255
            matrix = matrix.astype(np.uint8)
        else:
            matrix = matrix['hic']

            nums = int((matrix != 0).sum() / 1000)
            max_num = get_Several_MinMax_Array(matrix.reshape(-1), -nums)[0]
            matrix[matrix > max_num] = max_num
            matrix = matrix / matrix.max() * 255
            matrix = matrix.astype(np.uint8)

        matrix = merge_matrix(matrix)
        # matrix = np.triu(matrix).T + np.triu(matrix)
        
        # print("===> Generate String")

        fragments, interactions = [], []
        dim = matrix.shape
        # _fragments = calculate_fragments(chrom, resolution, matrix, np.zeros((dim[0], 5)))
        for i in range(dim[0]):
            _fragment = "{} 0 {} {} 0\n".format(chrom, i*resolution, int(matrix[i, i:i+2000].sum()*2))
            fragments.append(_fragment)
            for j in range(i, min(dim[0], i+2000)):
                if matrix[i, j] == 0:
                    continue
                _interaction = "{} {} {} {} {}\n".format(chrom, i*resolution, chrom, j*resolution, matrix[i, j])
                interactions.append(_interaction)
        
        # print("===> Write File")

        fragments_path = os.path.join(_out_dir, "fragments.gz")
        interactions_path = os.path.join(_out_dir, "interactions.gz")

        with gzip.open(fragments_path, 'wt') as file_object:
            file_object.writelines(fragments)
        with gzip.open(interactions_path, 'wt') as file_object:
            file_object.writelines(interactions)
        # with open(fragments_path, 'w') as file_object:
        #     file_object.writelines(fragments)
        # with open(interactions_path, 'w') as file_object:
        #     file_object.writelines(interactions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The script used to generate the fithic(or mustache) input file')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-i', dest='input_folder', help='', required=True)
    req_args.add_argument('-o', dest='output_folder', help='', required=True)

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-r', dest='resolution', type=int,
                           help='resolution(b)[default:1000]',
                           default=1000)
    misc_args.add_argument('-p', dest='is_predict', type=bool,
                           help='Whether it’s a predicted matrix or not',
                           default=False)

    args = parser.parse_args(sys.argv[1:])
    main(args)