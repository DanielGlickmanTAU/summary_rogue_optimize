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
        # print('input to robera shape' , input_ids_s_.shape)
        res = self.roberta(input_ids_s_, args['attention_mask_s'])
        # loss should be an nn module
        # print('original logits shape', res.logits.shape)
        logits = res.logits.view(-1)
        res['logits'] = logits
        res['logits'] = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
        if 'labels' in args:
            target = args['labels']
            # print('logits shape', logits.shape)
            # print('target shape', target.shape)
            assert target.shape == logits.shape
            loss = MSELoss()(input=logits, target=target)
            res['loss'] = loss
        # RE SHAPE EVAL INTO WHAT I NEED
        # if eval reshape labels into
        # print('shape', logits.shape)
        # print('shape labels', args['labels'].shape)
        # print('shape target and view', target.shape)
        return res
