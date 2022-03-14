# bigbed -> npy
import os
import sys
import argparse
import numpy as np
import pyBigWig
from epi_concat import mkdir, divide

def main(args):
    input_file = args.input_file
    output_folder = args.output_folder
    resolution = args.resolution
    subimage_size = args.subimage_size
    focus_size = args.focus_size
    print(args)
    if focus_size % subimage_size != 0:
        raise Exception()
    mkdir(output_folder)

    bb = pyBigWig.open(input_file)
    print(bb.isBigBed())
    print(bb.chroms())
    for chrom in bb.chroms():
        print(chrom)
        start, end = 0, bb.chroms(chrom)
        _matrix = np.zeros((round(end/resolution), round(end/resolution)), dtype=np.int8)
        for entry in bb.entries(chrom, start, end, withString=False):
            _x, _y = entry
            _x, _y = round(_x/resolution), round(_y/resolution)
            _matrix[_x, _y] = _matrix[_y, _x] = 1
        
        _matrix[_matrix==0] = -1
        _position = np.where(_matrix==1)
        for i in np.unique(_position[0]):
            _matrix[i] += 1
            _matrix[i][_matrix[i] > 1] = 1
        for j in np.unique(_position[1]):
            _matrix[:, j] += 1
            _matrix[:, j][_matrix[:, j] > 1] = 1
        _matrix[_matrix<0] = 0

        _matrix = divide(_matrix, focus_size, subimage_size)
        np.savez_compressed(os.path.join(output_folder, '{}_{}b'.format(chrom, resolution)), hic=_matrix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis annotation data')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-i', dest='input_file', help='', required=True)
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