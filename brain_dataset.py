import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BrainDataset(Dataset):
    def __init__(self, path, img_size=(256, 256), train=True):
        super(BrainDataset, self).__init__()

        self.train = train
        self.img_size = img_size
        self.dim = (3, img_size[0], img_size[1])

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.dim[-2], self.dim[-1]), Image.NEAREST),
            transforms.PILToTensor(),
        ])

        if self.train:
            x = np.load(os.path.join(path, f'X_0.npy'))
            x1 = np.load(os.path.join(path, 'X_no.npy'))
            x2 = np.load(os.path.join(path, 'X_an.npy'))
            if x.ndim == x1.ndim == x2.ndim:
                x = np.concatenate((x1, x2, x), axis=0)

            y = np.load(os.path.join(path, f'Y_no.npy'))
            y1 = np.load(os.path.join(path, 'Y_an.npy'))
            if y.ndim == y1.ndim:
                y = np.concatenate((y,y1), axis=0)

            gt = np.load(os.path.join(path, f'GT_no.npy'))
            gt1= np.load(os.path.join(path, f'GT_an.npy'))
            if gt.ndim == gt1.ndim:
                gt=np.concatenate((gt,gt1), axis=0)
        else:
            x = np.load(os.path.join(path, f'X_test.npy'))
            y = np.load(os.path.join(path, f'Y_test.npy'))
            gt = np.load(os.path.join(path, f'GT_test.npy'), allow_pickle=True)

        self.gt = gt
        self.labels = y
        self.images = x


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        if ( len(self.gt) >=1 ):
            image_label = self.gt[index]
            image = self.transform(image) / 255.0
            label = 0
            gt_label = self.transform(image_label) / 255.0
        else:
            image = self.transform(image) / 255.0
            label= -1
            gt_label = -1

        sample = {
            'image': image,
            'label': label,
            'gt_label': gt_label
        }
        return sample