import torch
from torch.utils.data import DataLoader
from lib.network import UNet
from torch import nn
from data_utils import TestDatasetNew
import argparse
from tqdm import tqdm
from torch.autograd import Variable
# from ConfusionMatrix import *
# torch.cuda.set_device(1)


def test():
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--save_img', default=0, type=int, help='')
    parser.add_argument('--crop_size', default=640, type=int, help='training images crop size') # 256 512 640 720 800

    parser.add_argument('--NetWeights', type=str, default='models_SameDevice/models_image_DiffDevice_natDD_512_stage2/net_epoch_300_0.145211_0.0000125_95.271.pth',
                        help="path of CSNet weights for testing")

    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size

    val_set = TestDatasetNew('./SelfData/DiffDevice/test/natDD', crop_size=CROP_SIZE)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=2, shuffle=False)

    net = UNet(output_dim=15)

    CrossEntropyLoss = nn.CrossEntropyLoss()

    if opt.NetWeights != '':
        checkpoint = torch.load(opt.NetWeights)
        net.load_state_dict(checkpoint['model_state_dict'])

    if torch.cuda.is_available():
        net.cuda()
        CrossEntropyLoss.cuda()

    # dict = {'D01': 1, 'D02': 2, 'D03': 3, 'D04': 4, 'D05': 5, 'D06': 6, 'D07': 7, 'D08': 8, 'D09': 9, 'D10': 10,
            # 'D11': 11, 'D12': 12, 'D13': 13, 'D14': 14, 'D15': 15, 'D16': 16, 'D17': 17, 'D18': 18, 'D19': 19,
            # 'D20': 20, 'D21': 21, 'D22': 22, 'D23': 23, 'D24': 24, 'D25': 25, 'D26': 26, 'D27': 27, 'D28': 28,
            # 'D29': 29, 'D30': 30, 'D31': 31, 'D32': 32, 'D33': 33, 'D34': 34, 'D35': 35}
    # label = [label for _, label in dict.items()]
    # confusion = ConfusionMatrix(num_classes=35, labels=label)

    for epoch in range(1, 1 + 1):
        train_bar = tqdm(val_loader)
        running_results = {'batch_sizes': 0, 'loss': 0, }

        net.eval()
        total_c = 0
        correct = 0

        for data, label in train_bar:
            batch_size = data.size(0)
            if batch_size <= 0:
                continue

            running_results['batch_sizes'] += batch_size

            label = Variable(label)
            if torch.cuda.is_available():
                label = label.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img, outputs = net(z)

            loss = CrossEntropyLoss(outputs, label)

            running_results['loss'] += loss.item() * batch_size

            _, predicted = outputs.max(1)
            total_c += label.size(0)
            correct += predicted.eq(label).sum().item()

            # confusion_matrix
            # confusion.update(predicted.cpu().numpy(), label.cpu().numpy())

            train_bar.set_description(desc='[%d] Loss_G: %.4f | Acc: %.3f%% ' % (
                epoch, running_results['loss'] / running_results['batch_sizes'], 100. * correct / total_c))

        print("averate acc is: ", 100. * correct / total_c)
        # confusion.plot()
        # confusion.summary()


if __name__ == '__main__':
    test()
