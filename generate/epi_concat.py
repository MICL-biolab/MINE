import os
import numpy as np
import numba as nb
from scipy import stats

in_dir = '/together/micl/liminghong/lab/epi'
out_dir = '/together/micl/liminghong/lab/train/epi_new'

ATAC_path = os.path.join(in_dir, 'ATAC')
DNase_path = os.path.join(in_dir, 'DNase')
Chip_H3K27ac_path = os.path.join(in_dir, 'Chip_H3K27ac')
Chip_H3K27me3_path = os.path.join(in_dir, 'Chip_H3K27me3')
Chip_H3K4me1_path = os.path.join(in_dir, 'Chip_H3K4me1')
all_path = [
    ATAC_path, DNase_path, Chip_H3K27ac_path, Chip_H3K27me3_path, Chip_H3K4me1_path
]

Max = 10000
resolution = 1000
# file_names = ['chr{}_{}b.npy'.format(i, resolution) for i in list(range(5, 23)) + ['X', 'Y']]
file_names = ['chr{}_{}b.npy'.format(i, resolution) for i in range(13, 23)]

@nb.jit()
def make_matrix(matrix, features):
    length = features.shape[0]
    for i in range(length):
        for j in range(i, length):
            # matrix[i, j] = stats.spearmanr(
            #     features[:, i].reshape(-1),
            #     features[:, j].reshape(-1)
            # )[0]

            max_num = min(
                features[i, :].max(), features[j, :].max()
            )
            if max_num == 0:
                matrix[i, j] = matrix[j, i] = Max
                continue
            _pearson = np.corrcoef(
                features[i, :], features[j, :]
            )
            matrix[i, j] = min(int(_pearson[0, 1] * Max), Max)
            matrix[j, i] = min(int(_pearson[1, 0] * Max), Max)
    return matrix
            

for file_name in file_names:
    # 构造特征矩阵
    epi_info = []
    for dir_path in all_path:
        file_path = os.path.join(dir_path, file_name)
        _epi = np.load(file_path)
        epi_info.append(_epi)
    epi_info = np.array(epi_info)
    epi_info = epi_info.T
    print(epi_info.shape)

    epi_matrix = np.zeros((epi_info.shape[0], epi_info.shape[0]), dtype=np.uint16)
    epi_matrix = make_matrix(epi_matrix, epi_info)

    prefix, ext = os.path.splitext(file_name)
    np.savez_compressed(os.path.join(out_dir, prefix), epi=epi_matrix)