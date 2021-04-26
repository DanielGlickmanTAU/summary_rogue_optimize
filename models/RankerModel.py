from dataclasses import dataclass

import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss


@dataclass
class Holder:
    data: object


class RankerModel(nn.Module):
    def __init__(self, roberta):
        super(RankerModel, self).__init__()
        self.roberta = roberta

    # def forward(
    #         self,
    #         input_ids=None,
    #         attention_mask=None,
    #         token_type_ids=None,
    #         position_ids=None,
    #         head_mask=None,
    #         inputs_embeds=None,
    #         labels=None,
    #         output_attentions=None,
    #         output_hidden_states=None,
    #         return_dict=None,
    # ):
    def forward(self, **args):
        input_ids_s_ = args['input_ids_s']  # shape(n_beam, tokenz_length)
        res = self.roberta(input_ids_s_, args['attention_mask_s'])

        logits = res.logits.view(-1)
        logits = (logits - logits.mean()) / (logits.std() + 0.01)
        res['logits'] = logits
        if 'labels' in args:
            target = args['labels']
            assert target.shape == logits.shape
            loss = MSELoss()(input=logits, target=target)
            res['loss'] = loss

        return res
