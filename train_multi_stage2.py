import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.network import UNet
from torch import nn
import os
import argparse
from tqdm import tqdm
from data_utils import TrainDatasetFromFolder_Multi
from torch.autograd import Variable
# torch.cuda.set_device(1)


def train():
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--crop_size', default=512, type=int, help='training images crop size')
    parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')

    parser.add_argument('--batchSize', default=10, type=int, help='train batch size')  # 22

    parser.add_argument('--loadEpoch', default=0, type=int, help='load epoch number')
    parser.add_argument('--generatorWeights', type=str, default='models_SameDevice/models_image_DiffDevice_natDD_512_stage1/net_epoch_300_0.154159_0.0000125.pth', help="path to CSNet weights (to continue training)")

    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    NUM_EPOCHS = opt.num_epochs
    LOAD_EPOCH = 0

    train_set = TrainDatasetFromFolder_Multi('./SelfData/DiffDevice/train/natDD', crop_size=CROP_SIZE)  # data2\train\natWA ./data/train/ours_3/train/nat
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=opt.batchSize, shuffle=True)

    net = UNet(output_dim=15)  # before unet

    mse_loss = nn.MSELoss()
    CrossEntropyLoss = nn.CrossEntropyLoss()

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    net.to(device)
    mse_loss.to(device)
    CrossEntropyLoss.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.0004, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)  # 60

    # checkpoint
    if opt.generatorWeights != '':

        checkpoint_finetune = torch.load(opt.generatorWeights)
        model_dict = net.module.state_dict()
        pretrained_dict = checkpoint_finetune['model_state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.module.load_state_dict(model_dict)
        net.eval()

    for epoch in range(LOAD_EPOCH, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'loss': 0, }

        net.train()
        scheduler.step()

        total_c = 0
        correct = 0

        for data, target, label in train_bar:
            batch_size = data.size(0)
            if batch_size <= 0:
                continue

            running_results['batch_sizes'] += batch_size

            real_img = Variable(target)
            label = Variable(label)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
                label = label.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img, outputs = net(z)

            # CS
            optimizer.zero_grad()

            loss_c = CrossEntropyLoss(outputs, label)

            loss = loss_c
            loss.backward()
            optimizer.step()

            running_results['loss'] += loss.item() * batch_size

            _, predicted = outputs.max(1)
            total_c += label.size(0)
            correct += predicted.eq(label).sum().item()

            train_bar.set_description(desc='[%d] Loss: %.4f lr: %.7f | Acc: %.3f%% ' % (
                epoch, running_results['loss'] / running_results['batch_sizes'],
                optimizer.param_groups[0]['lr'], 100. * correct / total_c))

        # for saving model
        save_dir = 'models_SameDevice/models_image_DiffDevice_natDD_512_stage2'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if epoch % 10 == 0:  # net.module.state_dict()

            all_states = {"model_state_dict": net.module.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
            # torch.save(obj=all_states, f=save_dir + '/epoch_' + str(epoch) + ',Loss_%6f'%
            #                              (running_results['loss'] / running_results['batch_sizes']) + ',Lr_%.7f'%
            #                              optimizer.param_groups[0]['lr'] + ',Acc_%.3f'%(100. * correct / total_c)
            #                              + ".pth")
            torch.save(obj=all_states, f=save_dir + '/net_epoch_%d_%6f_%.7f_%.3f.pth' % (
                epoch, running_results['loss'] / running_results['batch_sizes'], optimizer.param_groups[0]['lr'],
                100. * correct / total_c))


if __name__ == '__main__':
    train()
