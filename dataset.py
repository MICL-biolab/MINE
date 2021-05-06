import os
import glob
import torch
import torch.utils.data as data
import numpy as np
import numba as nb

# @nb.jit()
# def insert_data(hr_dataset, replaced_dataset, hr_matrix, replaced_matrix):
#     shape = hr_matrix.shape
#     for i in range(shape[0]):
#         for j in range(shape[1]):
#             hr_dataset.append(hr_matrix[i, j].astype(np.float32))
#             replaced_dataset.append(replaced_matrix[i, j].astype(np.float32))
#     return hr_dataset, replaced_dataset

class Dataset(data.Dataset):
    def __init__(self, data_path, chromosomes, resolution):
        super(Dataset, self).__init__()

        self.hr_dataset = None
        self.replaced_dataset = None
        for chromosome in chromosomes:
            file_name = '{}_{}.npz'.format(chromosome, resolution)
            hr_file = os.path.join(data_path, 'hr', file_name)
            replaced_file = os.path.join(data_path, 'replaced', file_name)

            hr_matrix = np.load(hr_file)['hic'].astype(np.float32)
            replaced_matrix = np.load(replaced_file)['hic'].astype(np.float32)
            max_num = 255
            hr_matrix[hr_matrix > max_num] = max_num
            replaced_matrix[replaced_matrix > max_num] = max_num
            self.shape = hr_matrix.shape
            
            hr_dataset = hr_matrix.reshape((
                self.shape[0] * self.shape[1], self.shape[2], self.shape[3]))
            replaced_dataset = replaced_matrix.reshape((
                self.shape[0] * self.shape[1], self.shape[2], self.shape[3]))

            if self.hr_dataset is None:
                self.hr_dataset = hr_dataset
                self.replaced_dataset = replaced_dataset
            else:
                self.hr_dataset = np.concatenate((self.hr_dataset, hr_dataset))
                self.replaced_dataset = np.concatenate((self.replaced_dataset, replaced_dataset))

            # self.hr_dataset, self.replaced_dataset = insert_data(
            #     self.hr_dataset, self.replaced_dataset, hr_matrix, replaced_matrix)

            # for i in range(self.shape[0]):
            #     for j in range(self.shape[1]):
            #         self.hr_dataset.append(
            #             hr_matrix[i, j].astype(np.float32))
            #         self.replaced_dataset.append(
            #             replaced_matrix[i, j].astype(np.float32))
        
        self.hr_dataset = torch.tensor(self.hr_dataset)
        self.replaced_dataset = torch.tensor(self.replaced_dataset)
    
    def __getitem__(self, index):
        input_tensor = torch.tensor(self.replaced_dataset[index])
        output_tensor = torch.tensor(self.hr_dataset[index])
        return input_tensor, output_tensor
        
    def __len__(self):
        return self.hr_dataset.shape[0]
