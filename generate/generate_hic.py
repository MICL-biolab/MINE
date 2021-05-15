import os
import sys
import gc
import time
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import traceback
from scipy.sparse import coo_matrix

res_map = {
    '1kb': 1_000, '5kb': 5_000, '10kb': 10_000, '25kb': 25_000,
    '50kb': 50_000, '100kb': 100_000, '250kb': 250_000, '500kb': 500_000,
    '1mb': 1_000_000
}

def mkdir(out_dir):
    if not os.path.isdir(out_dir):
        print(f'Making directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)


def readcoo2mat(cooFile, normFile, resolution):
    """function used for read a coordinated tag file to a square matrix."""
    norm = open(normFile, 'r').readlines()
    norm = np.array(list(map(float, norm)))
    compact_idx = list(np.where(np.isnan(norm)^True)[0])
    pd_mat = pd.read_csv(cooFile, sep='\t', header=None, dtype=int)
    row = pd_mat[0].values // resolution
    col = pd_mat[1].values // resolution
    val = pd_mat[2].values
    mat = coo_matrix((val, (row, col)), shape=(len(norm), len(norm)), dtype=np.uint16).toarray()
    # 归一化
    norm[np.isnan(norm)] = 1
    rows, cols = mat.shape
    max_num = 65535
    for i in range(rows):
        array = mat[i].astype(float) / norm / norm[i]
        array[array > max_num] = max_num
        mat[i] = array
    
    return mat, compact_idx

def read_data(data_file, norm_file, out_dir, resolution):
    filename = os.path.basename(data_file).split('.')[0] + '.npz'
    out_file = os.path.join(out_dir, filename)
    try:
        HiC, idx = readcoo2mat(data_file, norm_file, resolution)
    except:
        print(f'Abnormal file: {norm_file}')
        traceback.print_exc()
        raise Exception()
    np.savez_compressed(out_file, hic=HiC, compact=idx)
    print('Saving file:', out_file)

def main(args):
    raw_dir = args.input_folder
    out_dir = args.output_folder
    resolution = args.high_res
    map_quality = args.map_quality
    postfix = [args.norm_file, 'RAWobserved']

    # pool_num = min(23, multiprocessing.cpu_count())

    norm_files = []
    data_files = []
    for root, dirs, files in os.walk(raw_dir):
        if len(files) > 0:
            if (resolution in root) and (map_quality in root):
                for f in files:
                    if (f.endswith(postfix[0])):
                        norm_files.append(os.path.join(root, f))
                    elif (f.endswith(postfix[1])):
                        data_files.append(os.path.join(root, f))

    mkdir(out_dir)
    print(f'Start reading data, there are {len(norm_files)} files ({resolution}).')
    print(f'Output directory: {out_dir}')

    start = time.time()
    # pool = multiprocessing.Pool(processes=pool_num)
    # print(f'Start a multiprocess pool with process_num={pool_num} for reading raw data')
    # for data_fn, norm_fn in zip(data_files, norm_files):
    #     pool.apply_async(read_data, (data_fn, norm_fn, out_dir, res_map[resolution]))
    # pool.close()
    # pool.join()
    for i in range(len(norm_files)):
        read_data(data_files[i], norm_files[i], out_dir, res_map[resolution])
    print(f'All reading processes done. Running cost is {(time.time()-start)/60:.1f} min.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read raw data')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-i', dest='input_folder', help='', required=True)
    req_args.add_argument('-o', dest='output_folder', help='', required=True)

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-hr', dest='high_res', help='High resolution specified[default:1kb]',
                           default='1kb', choices=res_map.keys())
    misc_args.add_argument('-q', dest='map_quality', help='Mapping quality of raw data[default:MAPQGE30]',
                           default='MAPQGE30', choices=['MAPQGE30', 'MAPQG0'])
    misc_args.add_argument('-n', dest='norm_file', help='The normalization file for raw data[default:KRnorm]',
                           default='KRnorm', choices=['KRnorm', 'SQRTVCnorm', 'VCnorm'])
    
    args = parser.parse_args(sys.argv[1:])
    main(args)