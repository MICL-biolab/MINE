import os
import sys
import logging
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from train.dataset import hic_norm, get_Several_MinMax_Array

def merge_matrix(matrixs):
    dim = matrixs.shape
    result_map = np.zeros((dim[0]*dim[2], dim[0]*dim[2]), dtype=matrixs.dtype)
    for i in range(dim[0]):
        _r = result_map[int(i*dim[2]):int((i+1)*dim[2]), int(i*dim[2]):]
        _merge = np.hstack(matrixs[i])
        _y = min(_r.shape[0], _merge.shape[0])
        _x = min(_r.shape[1], _merge.shape[1])
        _r[:_y, :_x] = _merge[:_y, :_x]
    return result_map

def clean_matrix(matrixs):
    Min = matrixs[0].shape[0]
    for matrix in matrixs:
        Min = min(Min, matrix.shape[0])
    for matrix in matrixs:
        matrixs[0] = matrixs[0][:Min, :Min]
    return matrixs, Min


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def mkdir(out_dir):
    if not os.path.isdir(out_dir):
        print(f'Making directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)