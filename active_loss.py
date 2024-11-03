import torch
import torch.nn as nn
import numpy as np

class AAEXAD_loss(nn.Module):

        def __init__(self, lambda_u, lambda_n, lambda_a, f, cuda):
            super().__init__()
            self.lambda_u = lambda_u
            self.lambda_n = lambda_n
            self.lambda_a = lambda_a
            self.f = f
            self.use_cuda = cuda
        '''
        def forward(self, input, target, gt, y):

            print("y: ", y)

            rec_unlabeled = torch.where(y == 0, torch.sum((input - target) ** 2, dim=(1, 2, 3)), 0)
            rec_normal = torch.where(y == 1, torch.sum((input - target) ** 2, dim=(1, 2, 3)), 0)
            rec_anomalous = torch.where(y == -1, torch.sum((self.f(target) - input) ** 2, dim=(1, 2, 3)), 0)

            print("recs: (u,n,a) ", rec_unlabeled, rec_normal, rec_anomalous)

            loss_unlabeled = self.lambda_u * rec_unlabeled
            loss_normal = self.lambda_n * rec_normal
            loss_anomalous = self.lambda_a * rec_anomalous

            print("loss: (u,n,a) ", loss_unlabeled, loss_normal, loss_anomalous)

            total_loss = torch.sum(loss_unlabeled + loss_normal + loss_anomalous)

            return total_loss
            '''
        def forward(self, input, target, gt, y):
            print("y: ", y)

            # Maschere per i campioni
            mask_unlabeled = (y == 0)
            mask_normal = (y == 1)
            mask_anomalous = (y == -1)

            # Calcolo delle ricostruzioni solo per gli elementi che soddisfano le condizioni
            rec_unlabeled = torch.sum((input - target) ** 2, dim=(1, 2, 3)) * mask_unlabeled
            rec_normal = torch.sum((input - target) ** 2, dim=(1, 2, 3)) * mask_normal
            rec_anomalous = torch.sum((1 - target - input) ** 2, dim=(1, 2, 3)) * mask_anomalous

            print("recs: (u,n,a) ", rec_unlabeled, rec_normal, rec_anomalous)

            # Calcolo della loss per ogni tipo
            loss_unlabeled = self.lambda_u * rec_unlabeled
            loss_normal = self.lambda_n * rec_normal
            loss_anomalous = self.lambda_a * rec_anomalous

            print("loss: (u,n,a) ", loss_unlabeled, loss_normal, loss_anomalous)

            # Somma totale della loss
            total_loss = torch.sum(loss_unlabeled + loss_normal + loss_anomalous)

            return total_loss
