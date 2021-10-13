import os
import glob
import torch
import torch.utils.data as data
import numpy as np
import numba as nb


def get_Several_MinMax_Array(np_arr, several):
    """
    获取numpy数值中最大或最小的几个数
    :param np_arr:  numpy数组
    :param several: 最大或最小的个数（负数代表求最大，正数代表求最小）
    :return:
        several_min_or_max: 结果数组
    """
    if several > 0:
        several_min_or_max = np_arr[np.argpartition(np_arr, several)[:several]]
    else:
        several_min_or_max = np_arr[np.argpartition(np_arr, several)[several:]]
    return several_min_or_max


def hic_norm(matrix):
    nums = int((matrix != 0).sum() / 1000)
    max_num = get_Several_MinMax_Array(matrix.reshape(-1), -nums)[0]
    matrix[matrix > max_num] = max_num
    matrix[matrix <= 10] = matrix[matrix <= 10] / 10 * np.log2(10)
    matrix[matrix > 10] = np.log2(matrix[matrix > 10])

    matrix = matrix / matrix.max() * 255
    return matrix


def get_matrix(data_path, dir_name, chromosome, partition):
    file_path = os.path.join(
        data_path, dir_name, '{}_1000b.npz'.format(chromosome))
    matrix = np.load(file_path)[partition].astype(np.float32)
    return matrix


class Dataset(data.Dataset):
    def __init__(self, data_path, chromosomes, is_validate=False):
        super(Dataset, self).__init__()

        self.train_datasets = []
        for chromosome in chromosomes:
            hr_matrix = get_matrix(data_path, 'hr', chromosome, 'hic')
            replaced_matrix = get_matrix(
                data_path, 'replaced', chromosome, 'hic')
            epi_matrix = get_matrix(data_path, 'epi', chromosome, 'epi')
            annotation_matrix = get_matrix(data_path, 'annotation', chromosome, 'hic')
            old_train_matrixs = [hr_matrix, replaced_matrix, epi_matrix, annotation_matrix]

            # if not is_validate:
            #     for i in range(len(old_train_matrixs)):
            #         old_train_matrixs[i] = old_train_matrixs[i][:, 0, np.newaxis]
                # hr_matrix = hr_matrix[:, 0:2]
                # replaced_matrix = replaced_matrix[:, 0:2]
                # epi_matrix = epi_matrix[:, 0:2]

            # 清理全为0的矩阵
            old_shape = old_train_matrixs[0].shape
            train_matrixs = [[], [], [], []]
            for i in range(old_shape[0]):
                for j in range(old_shape[1]):
                    # 如果子矩阵全为0
                    if (not np.any(old_train_matrixs[0][i, j])) or (old_train_matrixs[3][i, j]==0).all():
                        continue
                    tmp_matrix = old_train_matrixs[0][i, j][old_train_matrixs[3][i, j]!=0]
                    if (tmp_matrix==0).all():
                        continue
                    if (np.count_nonzero(tmp_matrix) / tmp_matrix.size) < 0.1:
                        continue
                    for k in range(len(train_matrixs)):
                        train_matrixs[k].append(old_train_matrixs[k][i, j])
            for i in range(len(train_matrixs)):
                train_matrixs[i] = np.array(train_matrixs[i]).reshape((-1, old_shape[2], old_shape[3]))

            self.shape = train_matrixs[0].shape
            for i in range(1, len(train_matrixs)):
                _y = max(self.shape[0] - train_matrixs[i].shape[0], 0)
                train_matrixs[i] = np.pad(
                    train_matrixs[i],
                    ((0, _y), (0, 0), (0, 0)),
                    'constant', constant_values=(0, 0)
                )
                train_matrixs[i] = train_matrixs[i][:self.shape[0]]

            for i in range(len(train_matrixs) - 1):
                train_matrixs[i] = hic_norm(train_matrixs[i])

            # np.save('input_show/hr_input', train_matrixs[0])
            # np.save('input_show/replaced_input', train_matrixs[1])
            # np.save('input_show/epi_input', train_matrixs[2])
            # np.save('input_show/annotation_input', train_matrixs[3])

            train_datasets = train_matrixs
            if not self.train_datasets:
                self.train_datasets = train_datasets
            else:
                for i in range(len(train_datasets)):
                    self.train_datasets[i] = np.concatenate((self.train_datasets[i], train_datasets[i]))

        for i in range(len(self.train_datasets)):
            self.train_datasets[i] = torch.tensor(self.train_datasets[i])

    def __getitem__(self, index):
        replaced_tensor = torch.as_tensor(self.train_datasets[1][index])
        epi_tensor = torch.as_tensor(self.train_datasets[2][index])
        annotation_tensor = torch.as_tensor(self.train_datasets[3][index])
        output_tensor = torch.as_tensor(self.train_datasets[0][index])
        return replaced_tensor, epi_tensor, annotation_tensor, output_tensor

    def __len__(self):
        return self.train_datasets[0].shape[0]
