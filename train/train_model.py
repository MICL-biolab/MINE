import os
import sys
import argparse
import logging
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from torchelie.loss import PerceptualLoss
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from dataset import Dataset
import my_net as UNet
dist.init_process_group(backend='nccl')

# 配置每个进程的gpu
local_rank = dist.get_rank()  # 也可以通过设置args.local_rank得到（见下文）
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
print(device)


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

batch_size = 8
lr = 0.0001

train_chromosomes = ['chr{}'.format(i) for i in range(1, 18)]
test_chromosomes = ['chr{}'.format(i) for i in range(18, 23)]


def mkdir(out_dir):
    if not os.path.isdir(out_dir):
        print(f'Making directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()  # 总进程数
    return rt


def train(args):
    data_path = args.input_folder
    out_dir_path = args.output_folder
    mkdir(out_dir_path)

    train_set = Dataset(data_path, train_chromosomes)
    data_sampler = DistributedSampler(train_set)
    data_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False, sampler=data_sampler)

    test_set = Dataset(data_path, test_chromosomes)
    test_data_sampler = DistributedSampler(test_set)
    test_data_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, sampler=test_data_sampler)

    Net = UNet.Unet(1, 1)
    # Net.initialize_weights()
    optimizer = optim.Adam(Net.parameters(), lr=lr)

    Net.to(device)
    Net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Net).to(device)
    Net = torch.nn.parallel.DistributedDataParallel(
        Net, device_ids=[local_rank], output_device=local_rank)
    scaler = torch.cuda.amp.GradScaler()

    L1_loss = nn.L1Loss(reduction='mean').to(device)
    layer_names = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'maxpool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'maxpool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
                'conv3_4', 'relu3_4', 'maxpool3',  # noqa: E131
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
                'conv4_4', 'relu4_4', 'maxpool4',  # noqa: E131
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
                'conv5_4', 'relu5_4',  # 'maxpool5'
    ]
    perceptual_loss = PerceptualLoss(layer_names, rescale=True).to(device)

    logger = get_logger('exp.log')
    for epoch in range(0, 15000):
        running_loss = 0.0
        Net.train()
        output_max = 0.0
        if local_rank != -1:
            data_loader.sampler.set_epoch(epoch)
            test_data_loader.sampler.set_epoch(epoch)
        for iteration, batch in enumerate(data_loader, 1):
            replaced = Variable(batch[0]).to(device).unsqueeze(1)
            epi = Variable(batch[1]).to(device).unsqueeze(1)
            annotation = Variable(batch[2]).to(device).unsqueeze(1)
            target = Variable(batch[3]).to(device).unsqueeze(1)

            optimizer.zero_grad()
            annotation_is_0, annotation_is_not_0 = annotation==0, annotation!=0
            with torch.cuda.amp.autocast():
                output = Net(replaced, epi)
                output_max = max(output_max, output.max().item())

                loss1 = L1_loss(output[annotation_is_not_0], target[annotation_is_not_0])
                _output, _target = output.clone().detach(), target.clone().detach()
                _output[annotation_is_0] = 0
                _target[annotation_is_0] = 0
                loss2 = perceptual_loss(_output, _target)

            scaler.scale(loss1).backward(retain_graph=True)
            scaler.scale(loss2).backward()

            scaler.step(optimizer)
            scaler.update()

            dist.barrier()
            loss1 = reduce_tensor(loss1.clone())
            loss2 = reduce_tensor(loss2.clone())

            running_loss += loss1.item() + loss2.item()
            # running_loss += loss2.item()
            if iteration % 10 == 0 and local_rank == 0:
                # print(str(loss1) + " " + str(loss2))
                log_str = "===> Epoch[{}]({}/{}): Loss: {:.10f}".format(
                    epoch, iteration, len(data_loader), running_loss/iteration)
                print(log_str)

        dist.barrier()
        if local_rank == 0:
            logger.info(str(output_max))
        # 测试
        test_loss = 0.0
        Net.eval()
        for iteration, batch in enumerate(test_data_loader, 1):
            replaced = Variable(batch[0]).to(device).unsqueeze(1)
            epi = Variable(batch[1]).to(device).unsqueeze(1)
            annotation = Variable(batch[2]).to(device).unsqueeze(1)
            target = Variable(batch[3]).to(device).unsqueeze(1)
            
            annotation_is_0, annotation_is_not_0 = annotation==0, annotation!=0
            with torch.cuda.amp.autocast():
                output = Net(replaced, epi)

                loss1 = L1_loss(output[annotation_is_not_0], target[annotation_is_not_0])
                _output, _target = output.clone().detach(), target.clone().detach()
                _output[annotation_is_0] = 0
                _target[annotation_is_0] = 0
                loss2 = perceptual_loss(_output, _target)

            dist.barrier()
            loss1 = reduce_tensor(loss1.clone())
            loss2 = reduce_tensor(loss2.clone())

            test_loss += loss1.item() + loss2.item()

        if local_rank == 0:
            log_str = "===> Epoch[{}]:\ttrain_loss: {:.10f}\ttest_loss: {:.10f}".format(
                epoch, running_loss/len(data_loader), test_loss / len(test_data_loader))
            logger.info(log_str)

            save_checkpoint(Net, epoch, out_dir_path)
        dist.barrier()


def save_checkpoint(model, epoch, out_dir_path):
    model_out_path = os.path.join(
        out_dir_path, 'model_epoch_{}.pth'.format(epoch))
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-i', dest='input_folder', help='', required=True)
    req_args.add_argument('-o', dest='output_folder', help='', required=True)
    req_args.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    train(args)
