import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
from dataset import get_matrix, hic_norm
import my_net as UNet

class Dataset(data.Dataset):
    def __init__(self, data_path, chromosomes):
        super(Dataset, self).__init__()

        self.train_datasets = []
        for chromosome in chromosomes:
            replaced_matrix = get_matrix(
                data_path, 'replaced', chromosome, 'hic')
            epi_matrix = get_matrix(data_path, 'epi', chromosome, 'epi')
            train_matrixs = [replaced_matrix, epi_matrix]

            self.shape = train_matrixs[0].shape
            for i in range(1, len(train_matrixs)):
                _y = max(self.shape[0] - train_matrixs[i].shape[0], 0)
                _x = max(self.shape[1] - train_matrixs[i].shape[1], 0)
                train_matrixs[i] = np.pad(
                    train_matrixs[i],
                    ((0, _y), (0, _x), (0, 0), (0, 0)),
                    'constant', constant_values=(0, 0)
                )
                train_matrixs[i] = train_matrixs[i][:self.shape[0], :self.shape[1]]

            for i in range(len(train_matrixs)):
                train_matrixs[i] = hic_norm(train_matrixs[i])
                train_matrixs[i] = train_matrixs[i].reshape((
                    self.shape[0] * self.shape[1], self.shape[2], self.shape[3]))

            train_datasets = train_matrixs
            if not self.train_datasets:
                self.train_datasets = train_datasets
            else:
                for i in range(len(train_datasets)):
                    self.train_datasets[i] = np.concatenate((self.train_datasets[i], train_datasets[i]))

        for i in range(len(self.train_datasets)):
            self.train_datasets[i] = torch.tensor(self.train_datasets[i])

    def __getitem__(self, index):
        replaced_tensor = torch.as_tensor(self.train_datasets[0][index])
        epi_tensor = torch.as_tensor(self.train_datasets[1][index])
        return replaced_tensor, epi_tensor

    def __len__(self):
        return self.train_datasets[0].shape[0]

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

parser = argparse.ArgumentParser(description="Evaluation Script")
parser.add_argument("--train_folder", default="/data1/lmh_data/MINE/GM12878_ATAC_H3K27ac_H3K4me3/analyse/IMR90_ATAC_H3K27ac_H3K4me3/use_data",
                    type=str, help="The training data folder")
parser.add_argument(
    "--model", default="/data1/lmh_data/MINE/GM12878_ATAC_H3K27ac_H3K4me3/checkpoint/model_epoch_27.pth", type=str, help="model path")
parser.add_argument("--results", default="/data1/lmh_data/MINE/GM12878_ATAC_H3K27ac_H3K4me3/analyse/IMR90_ATAC_H3K27ac_H3K4me3/validation",
                    type=str, help="Result save location")

validate_chromosomes = ['chr{}'.format(i) for i in range(1, 23)]

opt = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.nn.DataParallel(UNet.Unet(1, 1))
model.load_state_dict(torch.load(opt.model, map_location=str(device)))

if not os.path.exists(opt.results):
    os.makedirs(opt.results)

model.cuda()
model.eval()

for chromosome in validate_chromosomes:
    train_set = Dataset(opt.train_folder, [chromosome])
    data_loader = data.DataLoader(train_set, batch_size=1, shuffle=False)

    print("===> Validation {}".format(chromosome))
    _i, _j = 0, 0
    output_data = np.zeros(train_set.shape)
    for iteration, batch in enumerate(data_loader, 1):
        replaced, epi = Variable(batch[0]), Variable(batch[1])

        replaced, epi = replaced.cuda(), epi.cuda()
        output = model(replaced.unsqueeze(1), epi.unsqueeze(1))
        output = output.detach().cpu().numpy()

        output = output[0, 0]
        output_data[_i, _j] = output
        _j += 1
        if _j >= train_set.shape[1]:
            _j = 0
            _i += 1

    np.savez_compressed('{}/{}_1000b.npz'.format(opt.results, chromosome), out=output_data)

