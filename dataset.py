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
        several_min_or_max = np_arr[np.argpartition(np_arr,several)[:several]]
    else:
        several_min_or_max = np_arr[np.argpartition(np_arr, several)[several:]]
    return several_min_or_max

def my_norm(matrix):
    dim = matrix.shape
    nums = int(dim[2] * dim[3] / 100)
    for y in range(dim[0]):
        for x in range(dim[1]):
            min_num = get_Several_MinMax_Array(matrix[y, x].reshape(-1), nums)[nums-1]
            max_num = get_Several_MinMax_Array(matrix[y, x].reshape(-1), -nums)[0]
            matrix[y, x][matrix[y, x]<min_num] = min_num
            matrix[y, x][matrix[y, x]>max_num] = max_num
            if max_num - min_num != 0:
                matrix[y, x] = (matrix[y, x] - min_num) / (max_num - min_num)
    return matrix

class Dataset(data.Dataset):
    def __init__(self, data_path, chromosomes, resolution, is_validate=False):
        super(Dataset, self).__init__()

        self.hr_dataset = None
        self.replaced_dataset = None
        for chromosome in chromosomes:
            file_name = '{}_{}.npz'.format(chromosome, resolution)
            hr_file = os.path.join(data_path, 'hr', file_name)
            replaced_file = os.path.join(data_path, 'replaced', file_name)
            epi_file = os.path.join(data_path, 'epi_new', file_name)

            hr_matrix = np.load(hr_file)['hic'].astype(np.float32)
            replaced_matrix = np.load(replaced_file)['hic'].astype(np.float32)
            epi_matrix = np.load(epi_file)['epi'].astype(np.float32)
            if not is_validate:
                hr_matrix = hr_matrix[:, 0, np.newaxis]
                replaced_matrix = replaced_matrix[:, 0, np.newaxis]
                epi_matrix = epi_matrix[:, 0, np.newaxis]

            epi_matrix = my_norm(epi_matrix)

            max_num = 255
            hr_matrix[hr_matrix > max_num] = max_num
            replaced_matrix[replaced_matrix > max_num] = max_num
            self.shape = hr_matrix.shape
            _y = max(self.shape[0] - epi_matrix.shape[0], 0)
            _x = max(self.shape[1] - epi_matrix.shape[1], 0)
            epi_matrix = np.pad(
                epi_matrix, ((0,_y),(0,_x),(0,0),(0,0)), 'constant', constant_values=(0, 0))
            epi_matrix = epi_matrix[:self.shape[0], :self.shape[1]]

            
            hr_dataset = hr_matrix.reshape((
                self.shape[0] * self.shape[1], self.shape[2], self.shape[3]))
            replaced_dataset = replaced_matrix.reshape((
                self.shape[0] * self.shape[1], 1, self.shape[2], self.shape[3]))
            epi_dataset = epi_matrix.reshape((
                self.shape[0] * self.shape[1], 1, self.shape[2], self.shape[3]))
            # replaced_dataset = replaced_dataset * epi_dataset

            replaced_dataset = np.concatenate((replaced_dataset, epi_dataset), axis=1)

            if self.hr_dataset is None:
                self.hr_dataset = hr_dataset
                self.replaced_dataset = replaced_dataset
            else:
                self.hr_dataset = np.concatenate((self.hr_dataset, hr_dataset))
                self.replaced_dataset = np.concatenate((self.replaced_dataset, replaced_dataset))
        
        self.hr_dataset = torch.tensor(self.hr_dataset)
        self.replaced_dataset = torch.tensor(self.replaced_dataset)
    
    def __getitem__(self, index):
        input_tensor = torch.tensor(self.replaced_dataset[index])
        output_tensor = torch.tensor(self.hr_dataset[index])
        return input_tensor, output_tensor
        
    def __len__(self):
        return self.hr_dataset.shape[0]
