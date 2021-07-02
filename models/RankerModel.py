from utils import compute

torch = compute.get_torch()
import torch.nn as nn


class RankerModel(nn.Module):
    def __init__(self, roberta, config, loss_fn: nn.Module):
        super(RankerModel, self).__init__()
        self._config = config
        self.roberta = roberta
        self.loss = loss_fn
        print('loss fn', self.loss)

    def forward(
            self,
            input_ids_s,
            attention_mask_s,
            labels=None,

    ):
        if isinstance(input_ids_s, list):
            assert not self.training, 'added this since I had problem with inference, since there is no data collator to transfer lists into tensors. should effect inference mode'
            input_ids_s = torch.stack(input_ids_s)
            attention_mask_s = torch.stack(attention_mask_s)

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
