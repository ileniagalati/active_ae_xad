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

        def forward(self, input, target, gt, y):

            rec_unlabeled = torch.where(y == 0, torch.sum((input - target) ** 2, dim=(1, 2, 3)), 0)
            rec_normal = torch.where(y == 1, torch.sum((input - target) ** 2, dim=(1, 2, 3)), 0)
            rec_anomalous = torch.where(y == -1, torch.sum((self.f(target) - input) ** 2, dim=(1, 2, 3)), 0)

            loss_unlabeled = self.lambda_u * rec_unlabeled
            loss_normal = self.lambda_n * rec_normal
            loss_anomalous = self.lambda_a * rec_anomalous

            total_loss = torch.sum(loss_unlabeled + loss_normal + loss_anomalous)

            return total_loss
