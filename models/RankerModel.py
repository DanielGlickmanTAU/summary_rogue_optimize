import torch.nn as nn
import torch


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
        # print(args['input_ids_s'])
        # can already change name...
        res = self.roberta(args['input_ids_s'], args['attention_mask_s'])
        # print(res)
        logits = res.logits
        
        return res
