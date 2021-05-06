import os
import sys
import numpy as np
from torch.utils import data
import torch
import torch.optim as optim
from torch.autograd import Variable
from time import gmtime, strftime
import torch.nn as nn
from dataset import Dataset
import hicplus_model
import multi_modal_model

use_gpu = True
batch_size = 1

resolution = '1kb'
data_path = '/together/micl/liminghong/hic_data/train'
train_chromosomes = ['chr{}'.format(i) for i in range(10, 21)]
test_chromosomes = ['chr{}'.format(i) for i in range(21, 23)]
out_dir_path = '/together/micl/liminghong/hic_data/train/checkpoint_new'


def train():
    train_set = Dataset(data_path, train_chromosomes, resolution)
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)

    test_set = Dataset(data_path, test_chromosomes, resolution)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    Net = hicplus_model.Net()
    # Net = multi_modal_model.ResidualLearningNet()
    if use_gpu:
        Net = Net.cuda()

    # optimizer = optim.SGD(Net.parameters(), lr = 0.001)
    mse_loss = nn.MSELoss()
    consine_loss = nn.CosineSimilarity()
    ssim_loss = pytorch_ssim.SSIM()
    Net.train()
    lr = 0.0001
    optimizer = optim.Adam(Net.parameters(), lr=lr, weight_decay=1e-5)
    # optimizer = optim.SGD(Net.parameters(), lr=lr, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    log_writer = open("log.txt", "w")
    for epoch in range(0, 1500):
        running_loss = 0.0
        for iteration, batch in enumerate(data_loader, 1):
            data, target = Variable(batch[0]), Variable(batch[1])
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            # max_num = target.max().item()
            # if max_num <= 0.0001:
            #     continue
            # data, target = torch.clamp(data / max_num, 0, 1), target / max_num
            # output = torch.clamp(Net(data.unsqueeze(1)), 0, 1)
            # import pdb
            # pdb.set_trace()
            output = Net(data.unsqueeze(1))

            optimizer.zero_grad()

            loss1 = mse_loss(output, target.unsqueeze(1))
            # loss2 = 0
            # for i in range(output.shape[0]):
            #     loss2 += 1. - torch.mean(consine_loss(output[i, 0], target[i]))
            # loss2 = loss2 / output.shape[0]

            # loss3 = (1. - ssim_loss(output, target.unsqueeze(1)))

            # loss1.backward(retain_graph=True)
            # loss2.backward(retain_graph=True)
            # loss3.backward()
            loss1.backward()

            optimizer.step()
            # scheduler.step(loss1 + loss2)

            running_loss += loss1.item()
            # running_loss += loss1.item()
            if iteration % 1000 == 0:
                log_str = "===> Epoch[{}]({}/{}): Loss: {:.10f}".format(
                    epoch, iteration, len(data_loader), running_loss/iteration)
                log_writer.write(log_str + "\n")
                print(log_str)
        
        # 测试
        test_loss = 0.0
        for iteration, batch in enumerate(test_data_loader, 1):
            data, target = Variable(batch[0]), Variable(batch[1])
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            output = Net(data.unsqueeze(1))

            loss1 = mse_loss(output, target.unsqueeze(1))
            # loss2 = 0
            # for i in range(output.shape[0]):
            #     loss2 += 1. - torch.mean(consine_loss(output[i, 0], target[i]))
            # loss2 = loss2 / output.shape[0]
            # loss3 = pytorch_ssim.ssim(output, target.unsqueeze(1))

            # test_loss += loss2.item() + loss3.item()
            test_loss += loss1.item()
        print("test Loss: {:.10f}".format(
            test_loss / len(test_data_loader)
        ))

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