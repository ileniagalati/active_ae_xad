import math
import os

import torch.optim
import torch.nn as nn
import torch.nn.init as init
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
    def __init__(self, latent_dim, lambda_p, lambda_u, lambda_n, lambda_a, f, path, AE_type, batch_size=None, silent=False, use_cuda=True,
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
            self.criterion = AAEXAD_loss(lambda_p, lambda_u, lambda_n, lambda_a, f, self.cuda)
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
        outputs = []
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
            outputs.extend(output)

        scores = np.array(scores)
        heatmaps = np.array(heatmaps)
        outputs = np.array(outputs)

        #gtmaps = np.array(gtmaps)
        #labels = np.array(labels)
        return heatmaps, scores, gtmaps, labels, outputs

    def train(self, epochs, save_path='', restart_from_scratch=False, iteration=0):
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

        save_latest_weights = os.path.join(os.path.dirname(save_path), str(iteration), f'latest_model_weights_{iteration}.pt')
        print("save path: ", save_latest_weights)
        load_latest_weights = os.path.join(os.path.dirname(save_path), str(iteration-1),  f'latest_model_weights_{iteration-1}.pt')
        print("load path: ", load_latest_weights)

        if not restart_from_scratch and os.path.exists(load_latest_weights):
            self.model.load_state_dict(torch.load(load_latest_weights))
            print("starting training from last iteration model weights...")
        elif restart_from_scratch:
            print("starting training from scratch...")

        initial_lr = 0.001  # Learning rate iniziale delle iterazioni
        initial_lr_it = initial_lr * 0.5
        tot_epochs = epochs * 20
        # Definisci il learning rate in base all'iterazione
        if iteration < 0: #2
            # Prima iterazione: cosine decay
            def get_lr(epoch):
                # Cosine decay da 0.001 a 0.0001
                min_lr = initial_lr_it
                return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / epochs))
        else:
            # Iterazioni successive: learning rate fisso più basso
            def get_lr(epoch):
                return initial_lr #_it
                #min_lr = initial_lr_it * 0.1
                #return min_lr + 0.5 * (initial_lr_it - min_lr) * (1 + math.cos(math.pi * epoch / tot_epochs))

        self.model.train()
        max_norm = 1.0
        norm = []


        for epoch in range(epochs):

            current_lr = get_lr(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            if epoch == 0 or epoch == 9 or epoch > 490:
                print(f"Iteration: {iteration}, Epoch: {epoch}, Learning rate: {current_lr:.6f}")

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

                '''#clipping
                #total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in self.model.parameters() if p.grad is not None]))
                norm.append(total_norm)
                if i % 100 == 0:  # ogni 100 batch
                    #print(f"Gradient norm before clipping: {total_norm:.2f}, clipped to: {max_norm:.1f}")

                    recent_norms = torch.tensor(norm[-100:])  # ultime 100 norme
                    #print(f"Average recent norm: {recent_norms.mean():.2f}")'''

                self.optimizer.step()

                train_loss += loss.item()
                tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / (i + 1)))

            '''# Salva pesi intermedi solo se specificato
            if self.save_intermediate and (epoch + 1) % 50 == 0:
                torch.save(self.model.state_dict(), os.path.join(save_path, f'{name}_{epoch}.pt'))'''

        # Salva i pesi finali dell'iterazione di active learning
        torch.save(self.model.state_dict(), save_latest_weights)
        print(f"saving weights: {save_latest_weights}")

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(filename))

    def load_weights(self, filename):
        self.model.load_state_dict(torch.load(filename))


def launch(data_path, epochs, batch_size, latent_dim, lambda_p, lambda_u, lambda_n, lambda_a, f, AE_type, loss='aexad',
           save_intermediate=False, save_path='', use_cuda=True, dataset='mnist', restart_from_scratch=False, iteration=0):
    trainer = Trainer(latent_dim,lambda_p, lambda_u, lambda_n, lambda_a, f, data_path, AE_type, batch_size, loss=loss,
                      save_intermediate=save_intermediate, use_cuda=use_cuda, dataset=dataset)

    #summary(trainer.model, (3, 448, 448))

    start = time()
    trainer.train(epochs, save_path=save_path, restart_from_scratch=restart_from_scratch, iteration=iteration)
    tot_time = time() - start

    heatmaps, scores, gtmaps, labels, output = trainer.test()

    return heatmaps, scores, gtmaps, labels, tot_time, output



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