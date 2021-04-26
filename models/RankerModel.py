import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss


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
        input_ids_s_ = args['input_ids_s']
        res = self.roberta(input_ids_s_, args['attention_mask_s'])
        # loss should be an nn module
        if self.training:
            logits = res.logits
            loss = MSELoss()(input=logits.view(-1), target=args['labels'].view(-1))
            res['loss'] = loss

        return res
