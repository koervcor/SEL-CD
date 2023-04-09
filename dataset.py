import glob
import os
import cv2
import torch
import numpy as np
from PIL import Image

def default_loader(path):
    img = Image.open(path)
    img = img.resize((96,96))
    label = getLabel(np.array(img))
    return img.convert("RGB"), label

def getLabel(img, c=1):
    _, thr = cv2.threshold(img, np.mean(img) + np.std(img) * c, 1, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    label = np.zeros_like(img)
    for k in range(len(contours)):
        if cv2.contourArea(contours[k]) > 30:
            cv2.drawContours(label, contours, k, 1, -1)
    return cv2.resize(label, (47,47))

class Dataset:

    def __init__(self, image_root_path, data_transforms=None, image_format='jpeg'):
        self.data_transforms = data_transforms
        self.image_root_path = image_root_path
        self.image_format = image_format
        self.images = []
        self.labels = []
        classes_folders = os.listdir(self.image_root_path)
        for cls_folder in classes_folders:
            folder_path = os.path.join(self.image_root_path, cls_folder)
            if os.path.isdir(folder_path):
                images_path = os.path.join(folder_path, "*.{}".format(self.image_format))
                images = glob.glob(images_path)
                self.images.extend(images)
        for image_file in self.images:
            label_name = os.path.basename(os.path.dirname(image_file))
            self.labels.append(int(label_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_file = self.images[item]
        label_name = os.path.basename(os.path.dirname(image_file))
        image, mask = default_loader(image_file)
        if self.data_transforms is not None:
            image = self.data_transforms(image)
            mask = mask[:,:,None].transpose(2, 0, 1)
            mask = torch.from_numpy(mask.astype(np.float32))
        return image, torch.tensor(int(label_name)), mask



