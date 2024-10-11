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

        # Trasformazioni da applicare alle immagini
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size[0], self.img_size[1]), Image.NEAREST),
            transforms.PILToTensor(),
        ])

        # Percorso delle immagini di training e test
        if self.train:
            split = 'train'
        else:
            split = 'test'

        x = np.load(os.path.join(path, f'X_{split}.npy'))  # / 255.0)[:,:,:,0]
        y = np.load(os.path.join(path, f'Y_{split}.npy'))
        gt = np.load(os.path.join(path, f'GT_{split}.npy'), allow_pickle=True)

        # normal_data = x[y == 0]
        # outlier_data = x[y == 1]

        self.gt = gt
        self.labels = y
        self.images = x


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image_label = self.gt[index]  # Ground truth vuota

        sample = {
            'image': self.transform(image) / 255.0,  # Normalizza l'immagine
            'label': 0,  # Placeholder per le etichette
            'gt_label': self.transform(image_label) / 255.0  # Ground truth vuota
        }
        return sample