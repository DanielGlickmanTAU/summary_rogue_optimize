import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from models.MeanCenteredMSELoss import MeanCenteredMSELoss
from models.NormalizedMSELoss import NormalizedMSELoss
from models.RankNetLoss import RankNetLoss
from models.RankingLoss import RankingLoss


class RankerModel(nn.Module):
    def __init__(self, roberta, config, loss_fn: nn.Module = None):
        super(RankerModel, self).__init__()
        self._config = config
        self.roberta = roberta
        self.loss = loss_fn if loss_fn else \
            MeanCenteredMSELoss(reduction='sum')
        # RankNetLoss()
        # NormalizedMSELoss(reduction='sum')
        # MSELoss(reduction='sum')
        # RankingLoss(tolerance=0.1, reduction='sum')
        print('loss fn', self.loss)

    def forward(
            self,
            input_ids_s,
            attention_mask_s,
            labels=None,

    ):
        res = self.roberta(input_ids_s, attention_mask_s)

        if labels is not None:
            logits = res.logits.view(-1)
            assert labels.shape == logits.shape

            loss = self.loss(logits, labels)
            res['loss'] = loss

            self.print_logits(labels, logits, loss)

        return res

    def print_logits(self, labels, logits, loss):
        if self._config.print_logits:
            print('__' * 10)
            print('logits', logits)
            print('labels', labels)
            print('loss', loss)
            print('__' * 10)
