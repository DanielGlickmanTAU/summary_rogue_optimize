from dataclasses import dataclass

import torch.nn as nn
import torch


class RankingLoss(nn.Module):

    def __init__(self, tolerance=0.01, reduction='sum'):
        super(RankingLoss, self).__init__()
        self.reduction = reduction
        assert tolerance is not None
        self.tolerance = tolerance

    def forward(self, logits, labels):
        assert logits.shape == labels.shape
        if len(logits.shape) == 1:
            return self._forward_single(labels, logits)
        element_wise_loss = [self._forward_single(label, logit) for label, logit in
                             zip(labels.unbind(), logits.unbind())]
        loss = torch.stack(element_wise_loss).mean()
        return loss

    def _forward_single(self, labels, logits):
        assert logits.shape == labels.shape
        assert len(logits.shape) == 1  # assuming 2 vectors
        indices = labels.sort(descending=True).indices
        # sorted_logits = logits[indices]
        # differences = nn.Parameter(torch.tensor([])).to(logits.device)
        differences = []
        for i, high_index in enumerate(indices[:-1]):
            for _, low_index in enumerate(indices[i + 1:]):
                logit_high = logits[high_index]
                logit_low = logits[low_index]
                label_high = labels[high_index]
                label_low = labels[low_index]

                logit_diff = logit_high - logit_low
                label_diff = label_high - label_low
                if abs(label_diff) > self.tolerance:
                    differences.append(logit_diff)

        if len(differences) == 0:
            return torch.tensor(0., requires_grad=True)

        differences = torch.stack(differences)
        sigmoid_log_diffs = differences.sigmoid().log()
        if self.reduction == 'sum':
            return sigmoid_log_diffs.sum()
        if self.reduction == 'mean':
            return sigmoid_log_diffs.mean()
        return sigmoid_log_diffs
