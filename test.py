import os
import config
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from dataset import Dataset
from torchvision import models, transforms


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

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x.view(x.size(0), -1))
        return x

def test(model, dataset, dataset_L):
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for batch, (image, label, mask) in enumerate(tqdm(dataset)):
            image = image.cuda()
            label = label.cuda()
            output = model(image)
            pred = output.data.max(1)[1]
            total_acc += pred.eq(label.data.view_as(pred)).sum()
    return float(total_acc) / dataset_L




def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--dataset', default='MSTAR', help='dataset name: MSTAR, OpenSARShip')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='test batch size')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='test number workers')
    parser.add_argument('--direction', default='left', type=str,
                        help='the direction of occlusion: top, bottom, left, right, all')

    args = parser.parse_args()

    if args.dataset.upper() == 'MSTAR':
        cfg = config.MSTAR
    elif args.dataset.upper() == 'OPENSARSHIP':
        cfg = config.OpenSARShip
    else:
        raise NotImplementedError
    return args, cfg



if __name__ == '__main__':
    args, cfg = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()

    backbone = Net(cfg["num_classes"])
    model_dict = backbone.state_dict()
    pretrained_dict = torch.load(cfg["predict_model"], map_location='cuda:0')['net']
    backbone.load_state_dict(pretrained_dict)
    backbone.eval()
    model_ft = backbone.cuda()
    valid_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=cfg["mean"],std=cfg["std"])
    ])

    test_dataset = Dataset(cfg["test_image_path"], valid_transform, cfg["image_format"])
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )
    test_accuray = test(model_ft, test_dataloader, len(test_dataset.images))

    occ_dataset = []
    if args.direction == "all":
        occ_dataset = ["top", "bottom", "left", "right"]
    else:
        occ_dataset.append(args.direction)

    for direction in occ_dataset:
        occ_test_accuray = []
        occ_level = os.listdir(os.path.join(cfg['predict_image_path'], direction))
        occ_level.sort()
        occ_path = os.path.join(cfg['predict_image_path'], direction)
        for level in occ_level:
            dataset_level = Dataset(os.path.join(occ_path, level), valid_transform, cfg["image_format"])
            occ_dataloader = torch.utils.data.DataLoader(
                dataset_level, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
            )
            occ_accuray = test(model_ft, occ_dataloader, len(dataset_level.images))
            occ_test_accuray.append(occ_accuray)

        print("Occlusion direction {}".format(direction))        
        print("Occlusion 10% Accuray = {}, Occlusion 20% Accuray = {}, Occlusion 30% Accuray = {}, Occlusion 40% Accuray = {}, Occlusion 50% Accuray = {}".format(occ_test_accuray[0], occ_test_accuray[1], \
            occ_test_accuray[2], occ_test_accuray[3], occ_test_accuray[4]))
    print("TEST Original Accuray = {}".format(test_accuray))
