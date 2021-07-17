import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3, 4'
import sys
import argparse
import numpy as np
from torch.utils import data
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import scipy.stats
from dataset import Dataset
import my_net as UNet
import logging
from torchelie.loss import PerceptualLoss

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

use_gpu = True
batch_size = 4
lr = 0.0001

# resolution = '1kb'
# data_path = '/data1/lmh_data/lab/train'
train_chromosomes = ['chr{}'.format(i) for i in range(13, 20)]
# test_chromosomes = ['chr{}'.format(i) for i in range(19, 20)]
# out_dir_path = '/data1/lmh_data/lab/train/checkpoint_new'


def mkdir(out_dir):
    if not os.path.isdir(out_dir):
        print(f'Making directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)


def train(args):
    data_path = args.input_folder
    out_dir_path = args.output_folder
    mkdir(out_dir_path)

    full_dataset = Dataset(data_path, train_chromosomes)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # train_set = Dataset(data_path, train_chromosomes)
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # test_set = Dataset(data_path, test_chromosomes)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    Net = UNet.Unet(1, 1)
    if use_gpu:
        Net = Net.cuda()
    Net.initialize_weights()

    L1_loss = nn.L1Loss(reduction='mean')
    layer_names = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'maxpool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'maxpool2',
        # 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
                # 'conv3_4', 'relu3_4', 'maxpool3',  # noqa: E131
        # 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
                # 'conv4_4', 'relu4_4', 'maxpool4',  # noqa: E131
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
                'conv5_4', 'relu5_4',  # 'maxpool5'
    ]
    perceptual_loss = PerceptualLoss(layer_names)
    if use_gpu:
        L1_loss = L1_loss.cuda()
        perceptual_loss = perceptual_loss.cuda()
    optimizer = optim.Adam(Net.parameters(), lr=lr)

    logger = get_logger('exp.log')
    for epoch in range(0, 15000):
        running_loss = 0.0
        Net.train()
        output_max = 0.0
        for iteration, batch in enumerate(data_loader, 1):
            replaced, epi, target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
            if use_gpu:
                replaced, epi, target = replaced.cuda(), epi.cuda(), target.cuda()
            if replaced.max() == 0:
                continue
            output = Net(replaced.unsqueeze(1), epi.unsqueeze(1))
            output_max = max(output_max, output.max().item())

            optimizer.zero_grad()
            loss1 = L1_loss(output, target.unsqueeze(1))
            loss2 = perceptual_loss(output, target.unsqueeze(1))

            loss1.backward(retain_graph=True)
            loss2.backward()

            optimizer.step()

            running_loss += loss1.item() + loss2.item()
            if iteration % 10 == 0:
                log_str = "===> Epoch[{}]({}/{}): Loss: {:.10f}".format(
                    epoch, iteration, len(data_loader), running_loss/iteration)
                print(log_str)
        torch.cuda.empty_cache()
        print(output_max)
        # 测试
        test_loss, accuracy = 0.0, 0.0
        length = len(test_data_loader)
        _i, _j = 0, 0
        # target_data, output_data = np.zeros(test_set.shape), np.zeros(test_set.shape)
        Net.eval()
        for iteration, batch in enumerate(test_data_loader, 1):
            replaced, epi, target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
            if use_gpu:
                replaced, epi, target = replaced.cuda(), epi.cuda(), target.cuda()
            output = Net(replaced.unsqueeze(1), epi.unsqueeze(1))

            loss1 = L1_loss(output, target.unsqueeze(1))
            loss2 = perceptual_loss(output, target.unsqueeze(1))

            test_loss += loss1.item() + loss2.item()
            
            outputs = output.detach().cpu().numpy()
            targets = target.detach().cpu().numpy()
            dim = outputs.shape
            _accuracy = 0.0
            for i in range(dim[0]):
                _a = scipy.stats.spearmanr(
                    outputs[i, 0].reshape(-1), targets[i].reshape(-1))[0]
                _accuracy += 1 if np.isnan(_a) else _a
            accuracy += _accuracy / dim[0]
        accuracy /= len(test_data_loader)

        log_str = "===> Epoch[{}]:\ttrain_loss: {:.10f}\ttest_loss: {:.10f}\taccuracy: {:.10f}".format(
            epoch, running_loss/len(data_loader), test_loss / len(test_data_loader), accuracy)
        logger.info(log_str)

        save_checkpoint(Net, epoch, out_dir_path)


def save_checkpoint(model, epoch, out_dir_path):
    model_out_path = os.path.join(
        out_dir_path, 'model_epoch_{}.pth'.format(epoch))
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-i', dest='input_folder', help='', required=True)
    req_args.add_argument('-o', dest='output_folder', help='', required=True)

    args = parser.parse_args(sys.argv[1:])
    train(args)
