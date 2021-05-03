import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from models.RankingLoss import RankingLoss


class RankerModel(nn.Module):
    def __init__(self, roberta, ):
        super(RankerModel, self).__init__()
        self.roberta = roberta
        print('warning, turning off dropout for linear layer')
        self.roberta.classifier.dropout.p = 0.

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
            # loss = -RankingLoss(tolerance=0.05, reduction='sum')(logits, labels)
            loss = BCEWithLogitsLoss()(logits, labels)
            print('__' * 10)
            print('logits', logits)
            print('labels', labels)
            print('loss', loss)
            print('__' * 10)
            res['loss'] = loss

        return res
