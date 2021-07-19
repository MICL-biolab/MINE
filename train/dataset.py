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

def hic_norm(matrix):
    nums = int((matrix != 0).sum() / 1000)
    max_num = get_Several_MinMax_Array(matrix.reshape(-1), -nums)[0]
    matrix[matrix>max_num] = max_num
    matrix[matrix<=10] = matrix[matrix<=10] / 10 * np.log2(10)
    matrix[matrix>10] = np.log2(matrix[matrix>10])

    matrix = matrix / matrix.max() * 255
    return matrix

def my_norm(matrix):
    dim = matrix.shape
    matrix_T = np.zeros(dim).astype(np.float32)
    for y in range(dim[0]):
        for x in range(dim[1]):
            if matrix[y, x].max():
                matrix[y, x] = np.arctan(matrix[y, x]) * 2 / np.pi
            matrix_T[y, x] = matrix[y, x].T
    matrix = matrix / matrix.max() * 255
    matrix_T = matrix_T / matrix_T.max() * 255
    return [matrix, matrix_T]

def get_matrix(data_path, dir_name, chromosome, partition):
    file_path = os.path.join(data_path, dir_name, '{}_1000b.npz'.format(chromosome))
    matrix = np.load(file_path)[partition].astype(np.float32)
    return matrix

class Dataset(data.Dataset):
    def __init__(self, data_path, chromosomes, is_validate=False):
        super(Dataset, self).__init__()

        self.hr_dataset = None
        self.replaced_dataset = None
        for chromosome in chromosomes:  
            hr_matrix = get_matrix(data_path, 'hr', chromosome, 'hic')
            replaced_matrix = get_matrix(data_path, 'replaced', chromosome, 'hic')
            epi_matrix = get_matrix(data_path, 'epi', chromosome, 'epi')

            max_num = 255
            epi_matrix = epi_matrix / epi_matrix.max() * max_num

            if not is_validate:
                # hr_matrix = hr_matrix[:, 0, np.newaxis]
                # replaced_matrix = replaced_matrix[:, 0, np.newaxis]
                # epi_matrix = epi_matrix[:, 0, np.newaxis]
                hr_matrix = hr_matrix[:, 0:2]
                replaced_matrix = replaced_matrix[:, 0:2]
                epi_matrix = epi_matrix[:, 0:2]
            
            self.shape = hr_matrix.shape

            _y = max(self.shape[0] - epi_matrix.shape[0], 0)
            _x = max(self.shape[1] - epi_matrix.shape[1], 0)
            epi_matrix = np.pad(
                epi_matrix, ((0,_y),(0,_x),(0,0),(0,0)), 'constant', constant_values=(max_num, max_num))
            epi_matrix = epi_matrix[:self.shape[0], :self.shape[1]]

            hr_matrix = hic_norm(hr_matrix)
            replaced_matrix = hic_norm(replaced_matrix)
            
            np.save('input_show/hr_input', hr_matrix)
            np.save('input_show/replaced_input', replaced_matrix)
            np.save('input_show/epi_input', epi_matrix)

            hr_dataset = hr_matrix.reshape((
                self.shape[0] * self.shape[1], self.shape[2], self.shape[3]))
            replaced_dataset = replaced_matrix.reshape((
                self.shape[0] * self.shape[1], self.shape[2], self.shape[3]))

            epi_dataset = epi_matrix.reshape((
                self.shape[0] * self.shape[1], self.shape[2], self.shape[3]))

            if self.hr_dataset is None:
                self.hr_dataset = hr_dataset
                self.replaced_dataset = replaced_dataset
                self.epi_dataset = epi_dataset
            else:
                self.hr_dataset = np.concatenate((self.hr_dataset, hr_dataset))
                self.replaced_dataset = np.concatenate((self.replaced_dataset, replaced_dataset))
                self.epi_dataset = np.concatenate((self.epi_dataset, epi_dataset))
        
        self.hr_dataset = torch.tensor(self.hr_dataset)
        self.replaced_dataset = torch.tensor(self.replaced_dataset)
        self.epi_dataset = torch.tensor(self.epi_dataset)
    
    def __getitem__(self, index):
        replaced_tensor = torch.as_tensor(self.replaced_dataset[index])
        epi_tensor = torch.as_tensor(self.epi_dataset[index])
        output_tensor = torch.as_tensor(self.hr_dataset[index])
        return replaced_tensor, epi_tensor, output_tensor
        
    def __len__(self):
        return self.hr_dataset.shape[0]
