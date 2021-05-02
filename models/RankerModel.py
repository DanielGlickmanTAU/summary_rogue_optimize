from dataclasses import dataclass

import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss


class RankerModel(nn.Module):
    def __init__(self, roberta):
        super(RankerModel, self).__init__()
        self.roberta = roberta

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
            loss = MSELoss()(input=logits, target=labels)
            print('loss', loss)
            res['loss'] = loss

        return res
