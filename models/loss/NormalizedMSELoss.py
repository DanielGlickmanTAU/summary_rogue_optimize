import torch.nn as nn
from torch.nn import MSELoss


class NormalizedMSELoss(nn.Module):

    def __init__(self, reduction='mean', smoothing=0.0001):
        super(NormalizedMSELoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, labels):
        labels = (labels - labels.mean()) / (labels.std() + self.smoothing)
        return MSELoss(reduction=self.reduction)(input=logits, target=labels)
