import torch
import torch.nn as nn
import numpy as np

class AAEXAD_loss(nn.Module):

        def __init__(self, lambda_n, lambda_a, f, cuda):
            super().__init__()
            self.lambda_n = lambda_n
            self.lambda_a = lambda_a
            self.f = f
            self.use_cuda = cuda

        def forward(self, input, target):
            rec_normal = torch.sum((input - target) ** 2, dim=(1, 2, 3))

            rec_anomalous = torch.sum((self.f(target) - input) ** 2, dim=(1, 2, 3))


            loss_normal = self.lambda_n * rec_normal
            loss_anomalous = self.lambda_a * rec_anomalous
            total_loss = torch.sum(loss_normal + loss_anomalous)

            return total_loss

