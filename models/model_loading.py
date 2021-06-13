from models.loss import loss_factory
from utils import compute

torch = compute.get_torch()
from transformers import BartTokenizer, BartForConditionalGeneration, RobertaForSequenceClassification, RobertaTokenizer
from models import RankerModel

xsum_model_name = 'sshleifer/distilbart-xsum-12-3'
cnn_model_name = 'sshleifer/distilbart-cnn-12-3'
# ranker_model_name = 'roberta-large'
ranker_model_name = 'roberta-base'
bart_base_model_name = 'facebook/bart-base'


def get_bart_model_and_tokenizer_xsum():
    return _get_bart_based_model_and_tokenizer(xsum_model_name)


def get_bart_model_and_tokenizer_cnn():
    return _get_bart_based_model_and_tokenizer(cnn_model_name)


def get_bart_base_model_and_tokenizer():
    return _get_bart_based_model_and_tokenizer(bart_base_model_name)


def _get_bart_based_model_and_tokenizer(model_name, tokenizer_name=None):
    if tokenizer_name is None:
        tokenizer_name = model_name
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(tokenizer_name,
                                              force_bos_token_to_be_generated=True,
                                              use_fast=True)
    adjust_model(model, tokenizer)
    return model, tokenizer


def get_ranker_model_and_tokenizer(config):
    if config.use_dropout:
        model = RobertaForSequenceClassification.from_pretrained(ranker_model_name, num_labels=1)
    else:
        print('disabling dropout')
        model = RobertaForSequenceClassification.from_pretrained(ranker_model_name, num_labels=1,
                                                                 attention_probs_dropout_prob=0.,
                                                                 hidden_dropout_prob=0.)
    print('using model', ranker_model_name)
    tokenizer = RobertaTokenizer.from_pretrained(ranker_model_name, use_fast=True)
    adjust_model(model, tokenizer)
    loss = loss_factory.get_loss(config)
    return RankerModel.RankerModel(model, config, loss_fn=loss), tokenizer


def adjust_model(model, tokenizer):
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
