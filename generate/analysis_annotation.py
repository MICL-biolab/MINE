# bigbed -> npy
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
    resolution = args.resolution
    mkdir(output_folder)

    bb = pyBigWig.open(input_file)
    print(bb.isBigBed())
    print(bb.chroms())
    for chrom in bb.chroms():
        print(chrom)
        start, end = 0, bb.chroms(chrom)
        _matrix = np.zeros((round(end/resolution), round(end/resolution)), dtype=np.uint8)
        for entry in bb.entries(chrom, start, end, withString=False):
            _x, _y = entry
            _x, _y = round(_x/resolution), round(_y/resolution)
            _matrix[_x, _y] = _matrix[_y, _x] = 1
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
    
    args = parser.parse_args(sys.argv[1:])
    main(args)