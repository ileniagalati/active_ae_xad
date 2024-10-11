import torch
import torch.nn as nn
import numpy as np

class AAEXAD_loss(nn.Module):
    def __init__(self, lambda_p, lambda_s, f, lambda_u=1.0, lambda_n=1.0, lambda_a=1.0, cuda=False):
        super().__init__()
        self.lambda_p = lambda_p  # Peso per esempi anomali
        self.lambda_s = lambda_s  # Peso per esempi supervisionati
        self.lambda_u = lambda_u  # Peso per esempi non etichettati
        self.lambda_n = lambda_n  # Peso per esempi normali etichettati
        self.lambda_a = lambda_a  # Peso per esempi anomali etichettati
        self.f = f  # Funzione F- per amplificare l'errore sugli esempi anomali
        self.use_cuda = cuda

    def forward(self, input, target, gt, y):
        # Calcolo della ricostruzione normale (per esempi normali o non etichettati)
        rec_n = (input - target) ** 2
        # Calcolo della ricostruzione amplificata (per esempi anomali)
        rec_o = (self.f(target) - input) ** 2

        # Componenti della loss
        # 1. Esempi non etichettati (X \setminus Q)
        loss_unlabeled = (1 - gt) * rec_n  # gt == 0 per esempi non etichettati

        # 2. Esempi normali etichettati (Q+)
        loss_normal = gt * (y == 0) * rec_n  # y == 0 per esempi normali

        # 3. Esempi anomali etichettati (Q-)
        loss_anomaly = gt * (y == 1) * rec_o  # y == 1 per esempi anomali

        # Applichiamo i pesi per le diverse componenti della loss
        loss = self.lambda_u * loss_unlabeled + self.lambda_n * loss_normal + self.lambda_a * loss_anomaly

        # Calcolo del peso lambda_p se non specificato (simile a quello già implementato nel codice)
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

        # Somma delle componenti della loss lungo le dimensioni spaziali
        loss = torch.sum(loss, dim=(1, 2, 3))

        # Peso batch a seconda del valore di y (supervisionato o meno)
        lambda_vec = torch.where(y == 1, self.lambda_s, 1.0)
        weighted_loss = torch.sum(loss * lambda_vec)

        # Normalizzazione finale
        return weighted_loss / torch.sum(lambda_vec)
