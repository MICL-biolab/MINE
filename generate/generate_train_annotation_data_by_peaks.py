# bigbed -> npy
import os
import sys
import argparse
import numpy as np
import pyBigWig
from epi_concat import mkdir, divide

def read_peaks_file(path, _peaks: dict):
    with open(path)as f:
        for line in f:
            datas = line.strip().split()
            left, right = int(int(datas[1])/1000), int(int(datas[2])/1000)
            peaks = list(range(left, right+1))
            if datas[0] not in _peaks.keys():
                _peaks[datas[0]] = set()
            for i in peaks:
                _peaks[datas[0]].add(i)
    return _peaks

def main(args):
    input_files = args.input_files
    output_folder = args.output_folder
    resolution = args.resolution
    subimage_size = args.subimage_size
    focus_size = args.focus_size
    print(args)
    if focus_size % subimage_size != 0:
        raise Exception()
    mkdir(output_folder)

    _peaks = dict()
    for input_file in input_files.split(','):
        _peaks = read_peaks_file(input_file, _peaks)
    
    for chr in range(1, 23):
        print(chr)
        _peak = _peaks['chr{}'.format(chr)]
        end = max(_peak)
        _matrix = np.zeros((end+1, end+1), dtype=np.int8)

        for _p in _peak:
            _matrix[:, _p] += 1
            _matrix[_p, :] += 1
        _matrix[_matrix!=2] = 0
        _matrix[_matrix==2] = 1
        

        _matrix = divide(_matrix, focus_size, subimage_size)
        np.savez_compressed(os.path.join(output_folder, 'chr{}_{}b'.format(chr, resolution)), hic=_matrix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis annotation data')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-i', dest='input_files', help='', required=True)
    req_args.add_argument('-o', dest='output_folder', help='', required=True)

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-r', dest='resolution', type=int,
                           help='resolution(b)[default:1000]',
                           default=1000)
    misc_args.add_argument('-s', dest='subimage_size', type=int,
                           help='The size of the captured image[default:400]',
                           default=400)
    misc_args.add_argument('-f', dest='focus_size', type=int,
                           help='The size of the picture to follow[default:2000]',
                           default=2000)
    
    args = parser.parse_args(sys.argv[1:])
    main(args)