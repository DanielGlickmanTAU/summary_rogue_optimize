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
        self.roberta(**args)
        # self.roberta(torch.tensor(args['input_ids_s][i]
        # single example not batch: torch.tensor(args['input_ids_s'][0])
        # self.roberta(torch.tensor(args['input_ids_s'][0]).to(self.roberta.device))
        # deal with mask
