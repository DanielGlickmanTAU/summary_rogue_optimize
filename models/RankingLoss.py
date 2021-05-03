import torch.nn as nn
import torch

tolerance = 0.01
reduction = 'sum'


class RankingLoss(nn.Module):
    def forward(self, logits, labels):
        assert logits.shape == labels.shape
        assert len(logits.shape) == 1  # assuming 2 vectors

        indices = labels.sort(descending=True).indices
        sorted_logits = logits[indices]
        differences = nn.Parameter(torch.tensor([]))

        for i, logit_high in enumerate(sorted_logits[:-1]):
            for j, logit_low in enumerate(sorted_logits[i + 1:]):
                diff = logit_high - logit_low
                if abs(diff) > tolerance:
                    differences = torch.cat((differences, torch.tensor([diff])))

        return differences.sigmoid().log()
