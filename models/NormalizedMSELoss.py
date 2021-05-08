import torch.nn as nn
from torch.nn import MSELoss


class NormalizedMSELoss(nn.Module):

    def __init__(self, smoothing=0.0001, reduction='sum'):
        super(NormalizedMSELoss, self).__init__()
        self.reduction = reduction
        self.smoothing = smoothing

    def forward(self, logits, labels):
        labels = (labels - labels.mean()) / (labels.std() + self.tolerance)
        return MSELoss(reduction=self.reduction)(input=logits, target=labels)
