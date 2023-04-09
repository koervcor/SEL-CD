import os
import config
import torch
import random
import argparse
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
from tqdm import tqdm
from torchvision import models, transforms, datasets
from dataset import Dataset


class Net(nn.Module):
    def __init__(self, classes=10):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.layer4 = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.fc = nn.Linear(6272, classes, bias=True)
    
    def forward(self, x, mask, epoch):
        x = self.layer1(x)
        x = self.SEL(x, mask, epoch)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.CD(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x.view(x.size(0), -1))
        return x

    def SEL(self, x, mask, epoch=0):
        if not self.training:
            return x
        fea_mean = torch.mean(x, dim=1)
        fea_ae = fea_mean * mask.squeeze(1)
        x = x + torch.tile(fea_ae[:,None,:,:], (1,16,1,1)) * (1 + np.cos(np.pi*epoch/cfg["epoch"]))/2

        return x
    
    def CD(self, x, r_min=0, r_max=0.6):
        if not self.training:
            return x
        N, C, H, W = x.shape
        mask = torch.ones(N, C, H, W, dtype=torch.float32)
        occlusion_max = int(r_max * C)
        occlusion_min = 0
        for i in range(N):
            index = self.rand(self.rand(1, occlusion_min, occlusion_max), r_min, C)
            mask[i].index_fill_(0, index, 0.0)

        drop_feature = x * mask.cuda()
        return drop_feature

    def rand(self, size=1, low=0, up=1):
        return (torch.rand(size) * up + low).type(torch.long)

def train(model, loss_func, dataset, optimizer, epoch):
    model.train()
    batch_loss = 0
    item = 0
    for batch, (image, label, mask) in enumerate(tqdm(dataset)):
        image = image.cuda()
        label = label.cuda()
        mask = mask.cuda()
        optimizer.zero_grad()
        output = model(image, mask, epoch)
        _, pred = torch.max(output, 1)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()
        batch_loss += loss.data.item()
        item += 1
    return batch_loss / item


def valid(model, loss_func, dataset, epoch):
    model.eval()
    batch_loss = 0
    item = 0
    with torch.no_grad():
        for batch, (image, label, mask) in enumerate(tqdm(dataset)):
            image = image.cuda()
            label = label.cuda()
            mask = mask.cuda()
            output = model(image, mask, epoch)
            loss = loss_func(output, label)
            batch_loss += loss.data.item()
            item += 1
    return batch_loss / item


def test(model, dataset, dataset_L):
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for batch, (image, label, mask) in enumerate(tqdm(dataset)):
            image = image.cuda()
            label = label.cuda()
            mask = mask.cuda()
            output = model(image, mask, epoch=0)
            pred = output.data.max(1)[1]
            total_acc += pred.eq(label.data.view_as(pred)).sum()
    return float(total_acc) / dataset_L


def train_model(model, loss_func, optimizer, step_scheduler, num_epochs=100):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])
    
    train_dataset = Dataset(cfg["train_image_path"], train_transform, cfg["image_format"])
    valid_dataset = Dataset(cfg["valid_image_path"], valid_transform, cfg["image_format"])
    test_dataset = Dataset(cfg["test_image_path"], valid_transform, cfg["image_format"])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    start_epoch = 0
    best_acc = 0
    for epoch in range(start_epoch + 1, num_epochs):
        train_epoch_loss = train(model, loss_func, train_dataloader, optimizer, epoch)
        valid_epoch_loss = valid(model, loss_func, valid_dataloader, epoch)
        test_accuracy = test(model, test_dataloader, len(test_dataset.images))
        step_scheduler.step()
        if round(test_accuracy, 6) >= round(best_acc, 6):
            best_acc = test_accuracy
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
                }
            save_model_file = os.path.join(cfg["model_output_dir"], cfg["chkpt"])
            if not os.path.exists(cfg["model_output_dir"]):
                os.makedirs(cfg["model_output_dir"])
            torch.save(checkpoint, save_model_file)
        if epoch % 10 == 0:
            print("Epoch = {} Train Loss = {} Valid Loss = {}  Test Accuray = {} ".format(epoch, train_epoch_loss,
                                                                                          valid_epoch_loss,
                                                                                          test_accuracy * 100))

    
class Loss(nn.Module):
    """docstring for Loss"""
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion =  nn.CrossEntropyLoss()

    def forward(self, output, target):
        L_cls = self.criterion(output, target)
        return L_cls


def seed_torch(seed=1234):
        random.seed(seed)
        torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--dataset', default='MSTAR', help='dataset name: MSTAR, OpenSARShip')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='test batch size')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='test number workers')

    args = parser.parse_args()

    if args.dataset.upper() == 'MSTAR':
        cfg = config.MSTAR
    else:
        raise NotImplementedError
    return args, cfg

if __name__ == '__main__':
    args, cfg = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()

    seed_torch(args.seed)
    backbone = Net(cfg["num_classes"])
    model_ft = backbone.cuda()
    criterion = Loss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=cfg["lr"])
    scheduler = lr_scheduler.MultiStepLR(optimizer_ft, cfg["step_lr"], gamma=cfg["gamma"])
    train_model(model_ft, criterion, optimizer_ft, scheduler, cfg["epoch"])
