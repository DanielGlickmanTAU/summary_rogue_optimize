from dataclasses import dataclass

import torch.nn as nn
import torch


@dataclass
class RankingLoss(nn.Module):
    reduction: str = 'sum'
    tolerance: float = 0.01

    # def __init__(self, tolerance=0.01, reduction='sum'):
    #     super(RankingLoss, self).__init__()
    #     self.reduction = reduction
    #     self.tolerance = tolerance

    def forward(self, logits, labels):
        assert logits.shape == labels.shape
        assert len(logits.shape) == 1  # assuming 2 vectors

        indices = labels.sort(descending=True).indices
        # sorted_logits = logits[indices]
        differences = nn.Parameter(torch.tensor([])).to(logits.device)

        for i, high_index in enumerate(indices[:-1]):
            for _, low_index in enumerate(indices[i + 1:]):
                logit_high = logits[high_index]
                logit_low = logits[low_index]
                label_high = labels[high_index]
                label_low = labels[low_index]

                logit_diff = logit_high - logit_low
                label_diff = label_high - label_low
                if abs(label_diff) > self.tolerance:
                    differences = torch.cat((differences, torch.tensor([logit_diff], device=differences.device)))

        sigmoid_log_diffs = differences.sigmoid().log()
        if self.reduction == 'sum':
            return sigmoid_log_diffs.sum()
        if self.reduction == 'mean':
            return sigmoid_log_diffs.mean()
        return sigmoid_log_diffs
