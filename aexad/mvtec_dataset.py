import PIL.Image as Image

import numpy as np
import os
import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms

class MvtecAD(Dataset):
    def __init__(self, path, seed=29, train=True):
        super(MvtecAD).__init__()

        self.train = train
        self.seed = seed
        self.dim = (3, 256, 256)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.dim[-2], self.dim[-1]), Image.NEAREST),
            #transforms.ToTensor(),
            transforms.PILToTensor()
        ])

        if self.train:
            split = 'train'
            x = np.load(os.path.join(path, f'X_{split}.npy'))  # / 255.0)[:,:,:,0]
            y = np.load(os.path.join(path, f'Y_{split}.npy'))
            gt = np.load(os.path.join(path, f'GT_{split}.npy'))

            print("training on x: ", x.shape)
            print("training on y: ", y.shape)
            print("training on gt: ", gt.shape)


        else:
            split = 'test'
            x = np.load(os.path.join(path, f'X_{split}.npy'))  # / 255.0)[:,:,:,0]
            y = np.load(os.path.join(path, f'Y_{split}.npy'))
            gt = np.load(os.path.join(path, f'GT_{split}.npy'))

            print("testing on x: ", x.shape)
            print("testing on y: ", y.shape)
            print("testing on gt: ", gt.shape)



        # normal_data = x[y == 0]
        # outlier_data = x[y == 1]

        self.gt = gt
        self.labels = y
        self.images = x


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.array(self.images[index], dtype=np.uint8)
        image_label = np.array(self.gt[index], dtype=np.uint8)
        print("img shape: ", image.shape)

        sample = {'image': self.transform(image) / 255.0,
                  'label': self.labels[index],
                  'gt_label': self.transform(image_label)/ 255.0}
        return sample
