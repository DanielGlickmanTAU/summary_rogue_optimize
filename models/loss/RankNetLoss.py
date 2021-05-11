import torch
from torch import nn
import torch.nn.functional as F


class RankNetLoss(nn.Module):
    def pairwise_diff(self, inputs):
        batch_s_ij = torch.unsqueeze(inputs, dim=-1) - torch.unsqueeze(inputs, dim=0)
        return batch_s_ij

    def forward(self, logits, labels):
        assert logits.shape == labels.shape
        if len(logits.shape) == 1:
            return self._forward_single(labels, logits)
        element_wise_loss = [self._forward_single(label, logit) for label, logit in
                             zip(labels.unbind(), logits.unbind())]
        loss = torch.stack(element_wise_loss).mean()
        return loss

        return self._forward_single(labels, logits)

    def _forward_single(self, labels, logits):
        batch_s_ij = self.pairwise_diff(logits)
        batch_p_ij = 1.0 / (torch.exp(- batch_s_ij) + 1.0)
        batch_std_diffs = self.pairwise_diff(labels)
        batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)  # ensuring S_{ij} \in {-1, 0, 1}
        batch_std_p_ij = 0.5 * (1.0 + batch_Sij)
        # about reduction, both mean & sum would work, mean seems straightforward due to the fact that the number of pairs differs from query to query
        batch_loss = F.binary_cross_entropy_with_logits(input=torch.triu(batch_p_ij, diagonal=1),
                                                        target=torch.triu(batch_std_p_ij, diagonal=1), reduction='mean')
        return batch_loss
