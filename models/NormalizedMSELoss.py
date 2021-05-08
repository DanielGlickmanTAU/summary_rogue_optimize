from dataclasses import dataclass

import torch.nn as nn
import torch
from torch.nn import MSELoss


@dataclass
class NormalizedMSELoss(nn.Module):
    reduction: str
    smoothing: float = 0.0001

    def forward(self, logits, labels):
        labels = (labels - labels.mean()) / (labels.std() + self.tolerance)
        return MSELoss(reduction=self.reduction)(input=logits, target=labels)
