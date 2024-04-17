import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.network import UNet_stage1
from torch import nn
import os
import argparse
from tqdm import tqdm
from data_utils import TrainDatasetFromFolder_Multi
from torch.autograd import Variable
# torch.cuda.set_device(0)

def train():
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--crop_size', default=512, type=int, help='training images crop size')  # 256
    parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')

    parser.add_argument('--batchSize', default=12, type=int, help='train batch size')  # 24

    parser.add_argument('--loadEpoch', default=0, type=int, help='load epoch number')
    parser.add_argument('--generatorWeights', type=str, default='', help="path to CSNet weights (to continue training)")

    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    NUM_EPOCHS = opt.num_epochs
    LOAD_EPOCH = 0

    train_set = TrainDatasetFromFolder_Multi('./SelfData/DiffDevice/train/natDD', crop_size=CROP_SIZE)  # data2\train\natWA natFBL
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=opt.batchSize, shuffle=True)

    net = UNet_stage1()  # before unet

    mse_loss = nn.MSELoss()

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    net.to(device)
    mse_loss.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.0004, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)  # 30

    # checkpoint
    if opt.generatorWeights != '':
        checkpoint = torch.load(opt.generatorWeights)
        net.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        LOAD_EPOCH = int(checkpoint['epoch'])
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
            fake_img = net(z)

            # PRNU
            optimizer.zero_grad()
            g_loss = mse_loss(fake_img, real_img)

            g_loss.backward()
            optimizer.step()

            running_results['loss'] += g_loss.item() * batch_size

            train_bar.set_description(desc='[%d] Loss: %.4f lr: %.7f ' % (
                epoch, running_results['loss'] / running_results['batch_sizes'],
                optimizer.param_groups[0]['lr']))

        # for saving model
        save_dir = 'models_SameDevice/models_image_DiffDevice_natDD_512_stage1'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if epoch % 10 == 0:  # net.module.state_dict()
            all_states = {"model_state_dict": net.module.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
            torch.save(obj=all_states, f=save_dir + '/net_epoch_%d_%6f_%.7f.pth' % (
                epoch, running_results['loss'] / running_results['batch_sizes'], optimizer.param_groups[0]['lr']))


if __name__ == '__main__':
    train()
