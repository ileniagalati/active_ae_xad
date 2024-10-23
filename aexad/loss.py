import torch
import torch.nn as nn
import numpy as np

class AEXAD_loss(nn.Module):

    def __init__(self, lambda_p, lambda_s, f, cuda):
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_s = lambda_s
        self.f = f
        self.use_cuda = cuda

    def forward(self, input, target, gt, y):
        rec_n = (input - target) ** 2
        rec_o = (self.f(target) - input) ** 2

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

        loss_vec = (1 - gt) * rec_n + lambda_p * gt * rec_o

        # Peso calcolato a livello di batch per momento: lambda_s calcolato qui
        #print(loss_vec.shape)
        loss = torch.sum(loss_vec, dim=(1, 2, 3))
        #lambda_vec = torch.Tensor(np.where(y==1, self.lambda_s, 1.0))

        lambda_vec = torch.where(y == 1, self.lambda_s, 1.0)
        #weighted_loss = torch.sum(torch.sum(loss, dim=1) * lambda_vec)
        weighted_loss = torch.sum(loss * lambda_vec)
        return weighted_loss / torch.sum(lambda_vec)