import os
import sys
import argparse
import numpy as np
import numba as nb
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tools import merge_matrix, hic_norm, clean_matrix

@nb.jit(nopython=True)
def make_score1(matrix1, matrix2, Min):
    score = 0
    for i in range(Min):
        _s = np.corrcoef(matrix1[i], matrix2[i])[0, 1]
        if np.isnan(_s):
            _s = 1
        score += _s
    score /= Min
    return score

@nb.jit(nopython=True)
def make_score2(matrix1, matrix2, Min):
    score, count = 0, 0
    for i in range(0, Min, 1000):
        _max = min(i+1000, Min)
        _s = np.corrcoef(matrix1[i:_max, i:_max].flatten(), matrix2[i:_max, i:_max].flatten())[0, 1]
        if np.isnan(_s):
            _s = 1
        score += _s
        count += 1
    score /= count
    return score

@nb.jit(nopython=True)
def make_score3(matrix1, matrix2, Min):
    scores = []
    begin = 300
    for i in range(0, 100):
        # _s = np.corrcoef(matrix1[i:i+400, i:i+400].flatten(), matrix2[i:i+400, i:i+400].flatten())[0, 1]
        # _s = np.corrcoef(matrix1[i], matrix2[i])[0, 1]
        _s = np.corrcoef(matrix1[begin:begin+i, begin:begin+i].flatten(), matrix2[begin:begin+i, begin:begin+i].flatten())[0, 1]
        # _s = np.corrcoef(np.diagonal(matrix1)[begin:begin+i], np.diagonal(matrix2)[begin:begin+i])[0, 1]
        if np.isnan(_s):
            _s = 1
        scores.append(_s)
    return scores

def make_ssim_score(matrix1, matrix2, Min):
    scores = []
    length = 400
    for i in range(0, Min, length):
        _max = min(i+length, Min)
        _s = structural_similarity(matrix1[i:_max, i:_max], matrix2[i:_max, i:_max])
        scores.append(_s)
    return scores

def main(args):
    background = np.load(args.background_path)['hic'].astype(float)
    another = np.load(args.another_path)['hic'].astype(float)
    background = hic_norm(background)
    another = hic_norm(another)

    # print('merge_mmatrix')
    background = merge_matrix(background)
    another = merge_matrix(another)
    # print('clean_matrix')
    background, another, Min = clean_matrix(background, another)
    # print('make_score')
    if args.function is 'pearson':
        print(make_score3(background, another, Min))
    elif args.function == 'ssim':
        print(make_ssim_score(background, another, Min))
    elif args.function == 'psnr':
        print(peak_signal_noise_ratio(background[14000:14400, 14000:14400], another[14000:14400, 14000:14400], data_range=255))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-b', dest='background_path', help='', required=True)
    req_args.add_argument('-a', dest='another_path', help='', required=True)

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    req_args.add_argument('-f', dest='function', help='', default='pearson')

    args = parser.parse_args(sys.argv[1:])
    main(args)