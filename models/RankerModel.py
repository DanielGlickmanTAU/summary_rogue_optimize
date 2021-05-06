import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from models.RankNetLoss import RankNetLoss
from models.RankingLoss import RankingLoss


class RankerModel(nn.Module):
    def __init__(self, roberta, config, loss_fn: nn.Module = None):
        super(RankerModel, self).__init__()
        self._config = config
        self.roberta = roberta
        self.loss = loss_fn if loss_fn else \
            RankNetLoss()
        # RankingLoss(tolerance=0.05, reduction='sum')
        # MSELoss(reduction='sum')
        print('loss fn', loss_fn)


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
        # labels = (labels - labels.mean()) / (labels.std() + 0.01)
        # loss = MSELoss(reduction='sum')(input=logits, target=labels)
        # print('wanring doing miinus loss')
        loss = self.loss(logits, labels)
        # loss = RankNetLoss()(logits, labels)
        # labels = labels * 20
        # loss = -BCEWithLogitsLoss()(logits, labels)
        self.print_logits(labels, logits, loss)
        res['loss'] = loss

    return res


def print_logits(self, labels, logits, loss):
    if self._config.print_logits:
        print('__' * 10)
        print('logits', logits)
        print('labels', labels)
        print('loss', loss)
        print('__' * 10)
