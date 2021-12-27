import os
import sys
import numpy as np
import scipy.stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from tools import merge_matrix, hic_norm

def show(use_data_path, validate_path, cell_line):
    length = 100 # 100kb
    complete_y = np.zeros((23, length))
    result_y = np.zeros((23, length))
    for chr in range(1, 23):
        chr_file_name = 'chr{}_1000b.npz'.format(chr)
        hr_path = os.path.join(use_data_path, 'hr', chr_file_name)
        replaced_path = os.path.join(use_data_path, 'replaced', chr_file_name)
        result_path = os.path.join(validate_path, chr_file_name)

        hr = np.load(hr_path)['hic']
        replaced = np.load(replaced_path)['hic']
        result = np.load(result_path)['out']

        hr = hic_norm(hr)
        replaced = hic_norm(replaced)
        result[result>255] = 255
        hr = hr.astype(np.uint8)
        replaced = replaced.astype(np.uint8)
        result = result.astype(np.uint8)

        hr = merge_matrix(hr)
        hr = np.triu(hr).T + np.triu(hr)
        replaced = merge_matrix(replaced)
        replaced = np.triu(replaced).T + np.triu(replaced)
        result = merge_matrix(result)
        result = np.triu(result).T + np.triu(result)

        # 1kb PCC
        distance_all = [[0, chr]]
        dic_norm = {}
        for i in range(length):
            dic_norm[i]=[[], [], []]
        for i in range(len(distance_all)):
            for j in range(-length+1, length, 1):
                dis = distance_all[i][0] - j
                dic_norm[abs(dis)][0]+=hr.diagonal(offset=j).tolist()
                dic_norm[abs(dis)][1]+=replaced.diagonal(offset=j).tolist()
                dic_norm[abs(dis)][2]+=result.diagonal(offset=j).tolist()

        _complete_y, _result_y = [], []
        for i in range(length):
            _complete_y.append(scipy.stats.pearsonr(dic_norm[i][0], dic_norm[i][1])[0])
            _result_y.append(scipy.stats.pearsonr(dic_norm[i][0], dic_norm[i][2])[0])

        complete_y[chr-1] = np.array(_complete_y)
        result_y[chr-1] = np.array(_result_y)
    
    np.save('temp/{}_complete_y.npy'.format(cell_line), complete_y)
    np.save('temp/{}_result_y.npy'.format(cell_line), result_y)

folder_path = '/data1/lmh_data/MINE/GM12878_ATAC_H3K27ac_H3K4me3/analyse'
use_data_path = os.path.join(folder_path, 'GM12878_ATAC_H3K27ac_H3K4me3', 'use_data')
validate_path = os.path.join(folder_path, 'GM12878_ATAC_H3K27ac_H3K4me3', 'validation')
show(use_data_path, validate_path, 'GM12878')

folder_path = '/data1/lmh_data/MINE/GM12878_ATAC_H3K27ac_H3K4me3/analyse'
use_data_path = os.path.join(folder_path, 'IMR90_ATAC_H3K27ac_H3K4me3', 'use_data')
validate_path = os.path.join(folder_path, 'IMR90_ATAC_H3K27ac_H3K4me3', 'validation')
show(use_data_path, validate_path, 'IMR90')

folder_path = '/data1/lmh_data/MINE/GM12878_ATAC_H3K27ac_H3K4me3/analyse'
use_data_path = os.path.join(folder_path, 'K562_ATAC_H3K27ac_H3K4me3', 'use_data')
validate_path = os.path.join(folder_path, 'K562_ATAC_H3K27ac_H3K4me3', 'validation')
show(use_data_path, validate_path, 'K562')