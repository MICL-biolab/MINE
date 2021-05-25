import os
import sys
import numpy as np
from torch.utils import data
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import scipy.stats
from dataset import Dataset
import hicplus_model
import multi_modal_model
import pytorch_ssim
import UNet_test as UNet
# import UNet
import logging

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
batch_size = 2

resolution = '1kb'
data_path = '/together/micl/liminghong/lab/train'
train_chromosomes = ['chr{}'.format(i) for i in range(15, 19)]
test_chromosomes = ['chr{}'.format(i) for i in range(19, 20)]
out_dir_path = '/together/micl/liminghong/lab/train/checkpoint_new'


def train():
    train_set = Dataset(data_path, train_chromosomes, resolution)
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = Dataset(data_path, test_chromosomes, resolution)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # Net = hicplus_model.Net(1)
    Net = UNet.Unet(1, 1)
    # Net = torch.load(
    #     "/together/micl/liminghong/lab/train/checkpoint_epoch_1/model_epoch_19.pth",
    #     map_location=lambda storage, loc: storage)["model"]
    # Net = multi_modal_model.ResidualLearningNet()
    if use_gpu:
        Net = Net.cuda()

    # optimizer = optim.SGD(Net.parameters(), lr = 0.001)
    mse_loss = nn.MSELoss()
    L1_loss = nn.L1Loss(reduction='mean')
    consine_loss = nn.CosineEmbeddingLoss()
    SL1_loss = nn.SmoothL1Loss()
    ssim_loss = pytorch_ssim.SSIM()
    if use_gpu:
        mse_loss = mse_loss.cuda()
        L1_loss = L1_loss.cuda()
        consine_loss = consine_loss.cuda()
        SL1_loss = SL1_loss.cuda()
        ssim_loss = ssim_loss.cuda()
    lr = 0.00001
    optimizer = optim.Adam(Net.parameters(), lr=lr, weight_decay=0.00001)
    # optimizer = optim.Adam(Net.parameters(), lr=lr)
    # optimizer = optim.SGD(Net.parameters(), lr=lr, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=len(data_loader), verbose=True)

    logger = get_logger('exp.log')
    for epoch in range(0, 1500):
        running_loss = 0.0
        Net.train()
        for iteration, batch in enumerate(data_loader, 1):
            data, target = Variable(batch[0]), Variable(batch[1])
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            # import pdb
            # pdb.set_trace()
            if data[:, 0].max() == 0:
                continue
            # output = Net(data[:, 0].unsqueeze(1))
            # output = Net(data)
            output = Net(data[:, 0].unsqueeze(1), data[:, 1].unsqueeze(1))
            # output = Net(data[:, 1].unsqueeze(1), data[:, 0].unsqueeze(1))
            # output = Net(data[:, 1].unsqueeze(1))

            optimizer.zero_grad()
            loss1 = mse_loss(output, target.unsqueeze(1))
            # loss1 = L1_loss(output, target.unsqueeze(1))
            # loss1 = SL1_loss(output, target.unsqueeze(1))
            # consine_target = torch.ones(output.shape[0], output.shape[1], output.shape[2], 1).cuda()
            # loss2 = consine_loss(output, target.unsqueeze(1), consine_target)
            loss2 = SL1_loss(output, target.unsqueeze(1))

            # loss3 = (1. - ssim_loss(output, target.unsqueeze(1)))

            loss1.backward(retain_graph=True)
            # loss2.backward(retain_graph=True)
            # loss3.backward()
            loss2.backward()

            optimizer.step()
            # scheduler.step(1. - ssim_loss(output, target.unsqueeze(1)))

            # running_loss += loss1.item() + loss3.item()
            # running_loss += loss2.item() + loss3.item()
            running_loss += loss1.item() + loss2.item()
            # if iteration % 500 == 0:
            #     log_str = "===> Epoch[{}]({}/{}): Loss: {:.10f}".format(
            #         epoch, iteration, len(data_loader), running_loss/iteration)
            #     log_writer.write(log_str + "\n")
            #     print(log_str)
        
        # 测试
        test_loss, accuracy = 0.0, 0.0
        length = len(test_data_loader)
        _i, _j = 0, 0
        target_data, output_data = np.zeros(test_set.shape), np.zeros(test_set.shape)
        Net.eval()
        for iteration, batch in enumerate(test_data_loader, 1):
            data, target = Variable(batch[0]), Variable(batch[1])
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            # output = Net(data[:, 0].unsqueeze(1))
            # output = Net(data)
            output = Net(data[:, 0].unsqueeze(1), data[:, 1].unsqueeze(1))
            # output = Net(data[:, 1].unsqueeze(1), data[:, 0].unsqueeze(1))
            # output = Net(data[:, 1].unsqueeze(1))

            loss1 = mse_loss(output, target.unsqueeze(1))
            # loss1 = L1_loss(output, target.unsqueeze(1))
            # loss1 = SL1_loss(output, target.unsqueeze(1))
            # consine_target = torch.ones(output.shape[0], output.shape[1], output.shape[2], 1).cuda()
            # loss2 = consine_loss(output, target.unsqueeze(1), consine_target)
            # loss3 = pytorch_ssim.ssim(output, target.unsqueeze(1))
            loss2 = SL1_loss(output, target.unsqueeze(1))

            # test_loss += loss1.item() + loss3.item()
            # test_loss += loss2.item() + loss3.item()
            test_loss += loss1.item() + loss2.item()
            # spearman = scipy.stats.spearmanr(
            #     torch.flatten(output).detach().cpu().numpy(),
            #     torch.flatten(target).detach().cpu().numpy())[0]
            # if spearman is np.nan:
            #     length -= 1
            # else:
            #     accuracy += spearman
            output = output.detach().cpu().numpy()[0, 0]
            output_data[_i, _j] = output
            target_data[_i, _j] = target.cpu().numpy()[0]
            _j += 1
            if _j >= test_set.shape[1]:
                _j = 0
                _i += 1
        
        accuracy = scipy.stats.spearmanr(
            np.hstack(np.hstack(target_data.astype(np.uint16))).reshape(-1),
            np.hstack(np.hstack(output_data.astype(np.uint16))).reshape(-1))[0]
        # output_data = np.hstack(np.hstack(output_data))
        # target_data = np.hstack(np.hstack(target_data))
        # output_data = output_data.reshape((1, 1, output_data.shape[0], output_data.shape[1]))
        # target_data = target_data.reshape((1, 1, target_data.shape[0], target_data.shape[1]))
        # accuracy = ssim_loss(torch.tensor(output_data), torch.tensor(target_data))

        log_str = "===> Epoch[{}]:\ttrain_loss: {:.10f}\ttest_loss: {:.10f}\taccuracy: {:.10f}".format(
            epoch, running_loss/len(data_loader), test_loss / len(test_data_loader), accuracy)
        logger.info(log_str)

        # log_writer.write("{}\n".format(accuracy / length))
        # print(accuracy / length)

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
    train()
