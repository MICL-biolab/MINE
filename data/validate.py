import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
from dataset_test import ValidateDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description="Evaluation Script")
parser.add_argument("--replaced_path", default="/data1/lmh_data/MMSR_complete/analyse/low/low10/chr19_low10.npz", type=str, help="")
parser.add_argument("--epi_path", default="/data1/lmh_data/MMSR_complete/train/epi/chr19_1000b.npz", type=str, help="")
parser.add_argument("--model", default="model_epoch_1415.pth", type=str, help="model path")
parser.add_argument("--result_path", default="/data1/lmh_data/MMSR_complete/analyse/low/low10/low10_result.npz", type=str, help="Result save location")

use_gpu = True
resolution = '1kb'

opt = parser.parse_args()
model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

if use_gpu:
    model.cuda()
model.eval()

validate_set = ValidateDataset(opt.replaced_path, opt.epi_path)
data_loader = data.DataLoader(validate_set, batch_size=1, shuffle=False)

ssim, old_ssim = 0.0, 0.0
print("===> Validation")
_i, _j = 0, 0
output_data = np.zeros(validate_set.shape)
for iteration, batch in enumerate(data_loader, 1):
    replaced, epi = Variable(batch[0]), Variable(batch[1])
    if use_gpu:
        replaced, epi = replaced.cuda(), epi.cuda()
    output = model(replaced.unsqueeze(1), epi.unsqueeze(1))
    output = output.detach().cpu().numpy()

    output = output[0, 0]
    replaced = replaced.detach().cpu().numpy()[0]

    output_data[_i, _j] = output
    _j += 1
    if _j >= validate_set.shape[1]:
        _j = 0
        _i += 1

np.savez_compressed(opt.result_path, out=output_data)