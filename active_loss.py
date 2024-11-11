import torch
import torch.nn as nn
import numpy as np

class AAEXAD_loss(nn.Module):

        def __init__(self, lambda_p, lambda_u, lambda_n, lambda_a, f, cuda):
            super().__init__()
            self.lambda_u = lambda_u
            self.lambda_n = lambda_n
            self.lambda_a = lambda_a
            self.lambda_p = lambda_p
            self.f = f
            self.use_cuda = cuda

        '''def forward(self, input, target, gt, y):

            rec_unlabeled = torch.where(y == 0, torch.sum((target - input) ** 2, dim=(1, 2, 3)), 0)
            rec_normal = torch.where(y == 1, torch.sum((target - input) ** 2, dim=(1, 2, 3)), 0)
            rec_anomalous = torch.where(y == -1, torch.sum((self.f(input) - target) ** 2, dim=(1, 2, 3)), 0)

            loss_unlabeled = self.lambda_u * rec_unlabeled
            loss_normal = self.lambda_n * rec_normal
            loss_anomalous = self.lambda_a * rec_anomalous

            total_loss = torch.sum(loss_unlabeled + loss_normal + loss_anomalous)


            return total_loss'''

        def forward(self, input, target, gt, y):

            rec_unlabeled = torch.where(y == 0, torch.sum((input - target) ** 2, dim=(1, 2, 3)), 0)
            rec_normal = torch.where(y == 1, torch.sum((input - target) ** 2, dim=(1, 2, 3)), 0)

            if self.lambda_p is None:
                lambda_p = torch.reshape(np.prod(gt.shape[1:]) / torch.sum(gt, dim=(1, 2, 3)), (-1, 1))
                ones_v = torch.ones((gt.shape[0], np.prod(gt.shape[1:])))
                if self.use_cuda:
                    lambda_p = lambda_p.cuda()
                    ones_v = ones_v.cuda()
                lambda_p = ones_v * lambda_p
                lambda_p = torch.reshape(lambda_p, gt.shape)
                lambda_p = torch.where(gt == 1, lambda_p, gt)
            else:
                lambda_p = self.lambda_p
                if self.use_cuda:
                    lambda_p = lambda_p.cuda()

            rec_anomalous = torch.where(
                y == -1,
                torch.sum((1 - gt) * (input - target) ** 2 + lambda_p * gt * (self.f(target) - input) ** 2, dim=(1, 2, 3)),
                0
            )

            loss_unlabeled = self.lambda_u * rec_unlabeled
            loss_normal = self.lambda_n * rec_normal
            loss_anomalous = self.lambda_a * rec_anomalous

            total_loss = torch.sum(loss_unlabeled + loss_normal + loss_anomalous)

            return total_loss
