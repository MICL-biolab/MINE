import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
from skimage.measure import compare_ssim, compare_psnr
from skimage.metrics import structural_similarity
from dataset import Dataset

parser = argparse.ArgumentParser(description="Evaluation Script")
parser.add_argument("--model", default="/together/micl/liminghong/lab/train/checkpoint_new/model_epoch_3.pth", type=str, help="model path")
parser.add_argument("--results", default="/together/micl/liminghong/lab/train/validation", type=str, help="Result save location")

use_gpu = True
resolution = '1kb'
data_path = '/together/micl/liminghong/lab/train'
validate_chromosomes = ['chr{}'.format(19)]

opt = parser.parse_args()
model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

if not os.path.exists(opt.results):
    os.makedirs(opt.results)

if use_gpu:
    model.cuda()
model.eval()

train_set = Dataset(data_path, validate_chromosomes, resolution, is_validate=False)
data_loader = data.DataLoader(train_set, batch_size=1, shuffle=False)

ssim, old_ssim = 0.0, 0.0
print("===> Validation")
_i, _j = 0, 0
output_data = np.zeros(train_set.shape)
for iteration, batch in enumerate(data_loader, 1):
    data, target = Variable(batch[0]), Variable(batch[1])
    if use_gpu:
        data, target = data.cuda(), target.cuda()
    # output = model(data)
    # output = model(data[:, 0].unsqueeze(1))
    output = model(data[:, 0].unsqueeze(1), data[:, 1].unsqueeze(1))
    # output = model(data[:, 1].unsqueeze(1), data[:, 0].unsqueeze(1))
    # output = model(data[:, 1].unsqueeze(1))
    output = output.detach().cpu().numpy()

    output = output[0, 0]
    data = data.detach().cpu().numpy()[0][0]
    target = target.detach().cpu().numpy()[0]
    ssim += structural_similarity(output, target)
    old_ssim += structural_similarity(data, target)

    output_data[_i, _j] = output
    _j += 1
    if _j >= train_set.shape[1]:
        _j = 0
        _i += 1

    # basename = os.path.basename(h5pyfilename)[:-3]
    # image = h5py.File(h5pyfilename, 'r')['data'][:] / 65535.0
    # image_t = torch.from_numpy(image)
    
    # output = get_output(image_t, model)
    
    # output_g = np.copy(output)
    # output_g[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    # output_g[:, :, 1::3, 1::3] = image[:, :, 1::3, 1::3]
    
    # print("===> Image %d" % iteration)

    if iteration % 100 == 0:
        print('ssim: {}'.format(ssim / iteration))
        print('old_ssim: {}'.format(old_ssim / iteration))

np.savez_compressed('{}/{}_{}.npz'.format(opt.results, validate_chromosomes[0], resolution), out=output_data.astype(np.uint16))


def test():
    import numpy as np
    import scipy.stats
    hr = np.load('hr/chr19_1kb.npz')['hic']
    replaced = np.load('replaced/chr19_1kb.npz')['hic']
    out = np.load('validation/chr19_1kb.npz')['out']
    hr_matrix = np.hstack(np.hstack(hr))
    replaced_matrix = np.hstack(np.hstack(replaced))
    out_matrix = np.hstack(np.hstack(out.astype(np.uint16)))
    scipy.stats.spearmanr(hr_matrix.reshape(-1), out_matrix.reshape(-1))