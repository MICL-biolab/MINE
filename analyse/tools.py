import os
import numpy as np

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

def get_Several_MinMax_Array(np_arr, several):
    """
    获取numpy数值中最大或最小的几个数
    :param np_arr:  numpy数组
    :param several: 最大或最小的个数（负数代表求最大，正数代表求最小）
    :return:
        several_min_or_max: 结果数组
    """
    if several > 0:
        several_min_or_max = np_arr[np.argpartition(np_arr,several)[:several]]
    else:
        several_min_or_max = np_arr[np.argpartition(np_arr, several)[several:]]
    return several_min_or_max

def hic_norm(matrix):
    nums = int((matrix != 0).sum() / 1000)
    max_num = get_Several_MinMax_Array(matrix.reshape(-1), -nums)[0]
    matrix[matrix>max_num] = max_num
    matrix[matrix<=10] = matrix[matrix<=10] / 10 * np.log2(10)
    matrix[matrix>10] = np.log2(matrix[matrix>10])

    matrix = matrix / matrix.max() * 255
    return matrix

def clean_matrix(matrixs):
    # Min = min(matrix1.shape[0], matrix2.shape[0])
    # matrix1 = matrix1[:Min, :Min]
    # matrix2 = matrix2[:Min, :Min]
    Min = matrixs[0].shape[0]
    for matrix in matrixs:
        Min = min(Min, matrix.shape[0])
    for matrix in matrixs:
        matrixs[0] = matrixs[0][:Min, :Min]

    # _both = (matrix1 != 0) | (matrix2 != 0)
    # return matrix1[_both], matrix2[_both], Min
    return matrixs, Min