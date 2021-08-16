import os
import sys
import argparse
import numpy as np
import numba as nb
from tools import merge_matrix, hic_norm

@nb.jit()
def down_sample(matrix, ratio):
    dim = matrix.shape
    _l = int(dim[0]/ratio)
    _m = np.zeros((_l, _l))
    for i in range(_l):
        for j in range(_l):
            _m[i, j] = matrix[i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio].mean()
    return _m

def main(args):
    matrix_path = args.matrix_path
    out_dir = args.output_folder
    chrom = args.chrom
    resolution = args.resolution
    is_predict = args.is_predict
    downsample_radio = args.downsample_radio

    matrix = np.load(matrix_path)

    if is_predict is True:
        matrix = matrix['out']
        for i in range(matrix.shape[0]):
            matrix[i][0] = (matrix[i][0] + matrix[i][0].T)/2
    else:
        matrix = matrix['hic']

    matrix = merge_matrix(matrix)
    matrix = np.triu(matrix).T + np.triu(matrix)

    if downsample_radio > 1:
        matrix = down_sample(matrix, downsample_radio)

    if not is_predict:
        matrix = hic_norm(matrix)
    
    print("===> Generate String")

    fragments, interactions = [], []
    dim = matrix.shape
    matrix = matrix.astype(np.int)
    # import pdb
    # pdb.set_trace()
    for i in range(dim[0]):
        matrix[i][matrix[i]<10] = 0
        _fragment = "{} 0 {} {} 0\n".format(chrom, i*resolution, matrix[i].sum())
        fragments.append(_fragment)
        for j in range(i, dim[0]):
            if matrix[i, j] == 0:
                continue
            _interaction = "{} {} {} {} {}\n".format(chrom, i*resolution, chrom, j*resolution, matrix[i, j])
            interactions.append(_interaction)
    
    print("===> Write File")

    fragments_path = os.path.join(out_dir, "fragments")
    interactions_path = os.path.join(out_dir, "interactions")

    with open(fragments_path, 'w') as file_object:
        file_object.writelines(fragments)
    with open(interactions_path, 'w') as file_object:
        file_object.writelines(interactions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='用于生成fithic输入文件的脚本')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-i', dest='matrix_path', help='', required=True)
    req_args.add_argument('-o', dest='output_folder', help='', required=True)
    req_args.add_argument('-c', dest='chrom', type=int, help='', required=True)

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-r', dest='resolution', type=int,
                           help='resolution(b)[default:1000]',
                           default=1000)
    misc_args.add_argument('-p', dest='is_predict', type=bool,
                           help='Whether it’s a predicted matrix or not',
                           default=False)
    misc_args.add_argument('-d', dest='downsample_radio', type=int,
                           help='',
                           default=1)

    args = parser.parse_args(sys.argv[1:])
    main(args)