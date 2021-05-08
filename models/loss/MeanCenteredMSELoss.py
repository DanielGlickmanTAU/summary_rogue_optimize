import torch.nn as nn
from torch.nn import MSELoss


class MeanCenteredMSELoss(nn.Module):

    def __init__(self, reduction='sum'):
        super(MeanCenteredMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, labels):
        labels = (labels - labels.mean())
        return MSELoss(reduction=self.reduction)(input=logits, target=labels)
