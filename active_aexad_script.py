import os

import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from time import time
from torchsummary import summary

from aexad.AE_architectures import Shallow_Autoencoder, Deep_Autoencoder, Conv_Autoencoder, PCA_Autoencoder, Conv_Deep_Autoencoder, Conv_Autoencoder_f2
from aexad.dataset import CustomAD
from aexad.loss import AEXAD_loss
from active_loss import AAEXAD_loss

from scipy.ndimage import gaussian_filter

from aexad.mvtec_dataset import MvtecAD
from brain_dataset import BrainDataset

class Trainer:
    def __init__(self, latent_dim, lambda_u, lambda_n, lambda_a, f, path, AE_type, batch_size=None, silent=False, use_cuda=True,
                 loss='aexad', save_intermediate=False, dataset='mnist'):
        '''
        :param latent_dim:
        :param lambda_p: anomalous pixel weight
        :param lambda_s: anomalous samples weight
        :param f: mapping function for anomalous pixels
        :param path:
        :param AE_type:
        :param batch_size:
        :param silent:
        :param use_cuda:
        '''
        self.silent = silent

        if dataset == 'mnist' or dataset == 'fmnist':
            self.train_loader = DataLoader(CustomAD(path, train=True), batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(CustomAD(path, train=False), batch_size=batch_size, shuffle=False)
        if dataset == 'mvtec':
            self.train_loader = DataLoader(MvtecAD(path, train=True), batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(MvtecAD(path, train=False), batch_size=batch_size, shuffle=False)
        if dataset == 'brain':
            self.train_loader = DataLoader(BrainDataset(path, train=True), batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(BrainDataset(path, train=False), batch_size=batch_size, shuffle=False)

        self.save_intermediate = save_intermediate

        if lambda_a is None:
            lambda_a = len(self.train_loader.dataset) / np.sum(self.train_loader.dataset.labels)

        self.cuda = use_cuda and torch.cuda.is_available()

        if AE_type == 'shallow':
            self.model = Shallow_Autoencoder(self.train_loader.dataset.dim, np.prod(self.train_loader.dataset.dim),
                                             latent_dim)
        # deep
        elif AE_type == 'deep':
            self.model = Deep_Autoencoder(self.train_loader.dataset.dim, flat_dim=np.prod(self.train_loader.dataset.dim),
                                          intermediate_dim=256, latent_dim=latent_dim)
        # conv
        elif AE_type == 'conv':
            self.model = Conv_Autoencoder(self.train_loader.dataset.dim)

        elif AE_type == 'conv_deep':
            self.model = Conv_Deep_Autoencoder(self.train_loader.dataset.dim)

        elif AE_type == 'conv_f2':
            self.model = Conv_Autoencoder_f2(self.train_loader.dataset.dim)

        elif AE_type == 'pca':
            self.model = PCA_Autoencoder(np.prod(self.train_loader.dataset.dim), np.prod(self.train_loader.dataset.dim),
                                         latent_dim)
        else:
            raise Exception()

        self.optimizer = torch.optim.Adam(self.model.parameters())
        if loss == 'aexad':
            self.criterion = AEXAD_loss(lambda_n, lambda_a, f, self.cuda)
        if loss == 'aaexad':
            self.criterion = AAEXAD_loss(lambda_u, lambda_n, lambda_a, f, self.cuda)
        elif loss == 'mse':
            self.criterion = nn.MSELoss()

        if self.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()



    def test(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, disable=self.silent)
        shape = [0]
        shape.extend(self.test_loader.dataset.images.shape[1:])
        heatmaps = []
        scores = []
        gtmaps = []
        labels = []
        for i, sample in enumerate(tbar):
            image= sample['image']
            label = sample['label']
            gtmap = sample['gt_label']

            # TODO ricontrollare
            if self.cuda:
                image = image.cuda()
            output = self.model(image).detach().cpu().numpy()
            image = image.cpu().numpy()
            heatmap = ((image-output) ** 2)#.numpy()
            score = heatmap.reshape((image.shape[0], -1)).mean(axis=-1)
            heatmaps.extend(heatmap)
            scores.extend(score)
            gtmaps.extend(gtmap.detach().numpy())
            labels.extend(label.detach().numpy())
        scores = np.array(scores)
        heatmaps = np.array(heatmaps)
        #gtmaps = np.array(gtmaps)
        #labels = np.array(labels)
        return heatmaps, scores, gtmaps, labels

    def train(self, epochs, save_path='', restart_from_scratch=False):
        if isinstance(self.model, Conv_Autoencoder):
            name = 'model_conv'
        elif isinstance(self.model, Conv_Autoencoder_f2):
            name = 'model_conv_f2'
        elif isinstance(self.model, Deep_Autoencoder):
            name = 'model_deep'
        elif isinstance(self.model, Shallow_Autoencoder):
            name = 'model'
        elif isinstance(self.model, Conv_Deep_Autoencoder):
            name = 'model_conv_deep'

        latest_weights = os.path.join(save_path, 'latest_model_weights.pt')
        if not restart_from_scratch and os.path.exists(latest_weights):
            self.model.load_state_dict(torch.load(latest_weights))
            print("starting training from last iteration model weights...")
        elif restart_from_scratch:
            self.initialize_model_weights()
            print("starting training from scratch...")

        self.model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            tbar = tqdm(self.train_loader, disable=self.silent)
            for i, sample in enumerate(tbar):
                image = sample['image']
                label = sample['label']
                gtmap = sample['gt_label']

                if self.cuda:
                    image = image.cuda()
                    gtmap = gtmap.cuda()
                    label = label.cuda()

                output = self.model(image)
                loss = self.criterion(output, image, gtmap, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / (i + 1)))

            # Salva pesi intermedi solo se specificato
            if self.save_intermediate and (epoch + 1) % 50 == 0:
                torch.save(self.model.state_dict(), os.path.join(save_path, f'{name}_{epoch}.pt'))

        # Salva i pesi finali dell'iterazione di active learning
        torch.save(self.model.state_dict(), latest_weights)
        print(f"saving weights: {latest_weights}")


    def initialize_model_weights(self):
        for layer in self.model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(filename))

    def load_weights(self, filename):
        self.model.load_state_dict(torch.load(filename))


def launch(data_path, epochs, batch_size, latent_dim, lambda_u, lambda_n, lambda_a, f, AE_type, loss='aexad',
           save_intermediate=False, save_path='', use_cuda=True, dataset='mnist', restart_from_scratch=False):
    trainer = Trainer(latent_dim, lambda_u, lambda_n, lambda_a, f, data_path, AE_type, batch_size, loss=loss,
                      save_intermediate=save_intermediate, use_cuda=use_cuda, dataset=dataset)

    #summary(trainer.model, (3, 448, 448))

    start = time()
    trainer.train(epochs, save_path=save_path, restart_from_scratch=restart_from_scratch)
    tot_time = time() - start

    heatmaps, scores, gtmaps, labels = trainer.test()

    return heatmaps, scores, gtmaps, labels, tot_time



if __name__ == '__main__':
    def f(x):
        return 1-x

    #data_path = 'test_data/mnist/0/2'
    #info = data_path.split('/')
    #dataset = info[-2]
    #seed = info[-1]

    data_path = 'mvtec'

    start = time()

    trainer = Trainer(32, None, None, f, data_path,
                      'conv_deep', 4, dataset='brain_dataset')
    #summary(trainer.model, (3, 900, 900))
    trainer.train(1000)
    print('TIME ELAPSED: ', time()-start)
    heatmaps, scores, gtmaps, labels = trainer.test()

    #ret_path = os.path.join('test_results')

    #if not os.path.exists(ret_path):
    #    os.makedirs(ret_path)

    #np.save(open(os.path.join(ret_path, 'aexad_htmaps.npy'), 'wb'), heatmaps)
    #np.save(open(os.path.join(ret_path, 'aexad_scores.npy'), 'wb'), scores)

    #if not os.path.exists(os.path.join(ret_path, 'gt.npy')):
    #    np.save(open(os.path.join(ret_path, 'gt.npy'), 'wb'), gtmaps)

    #if not os.path.exists(os.path.join(ret_path, 'labels.npy')):
    #    np.save(open(os.path.join(ret_path, 'labels.npy'), 'wb'), labels)