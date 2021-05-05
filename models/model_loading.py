from utils import compute

torch = compute.get_torch()
from transformers import BartTokenizer, BartForConditionalGeneration, RobertaForSequenceClassification, RobertaTokenizer
from models import RankerModel

xsum_model_name = 'sshleifer/distilbart-xsum-12-3'
cnn_model_name = 'sshleifer/distilbart-cnn-12-3'
# ranker_model_name = 'roberta-large'
ranker_model_name = 'roberta-base'


def get_bart_model_and_tokenizer_xsum():
    model = BartForConditionalGeneration.from_pretrained(xsum_model_name)
    tokenizer = BartTokenizer.from_pretrained(xsum_model_name,
                                              force_bos_token_to_be_generated=True,
                                              use_fast=True)
    adjust_model(model, tokenizer)
    return model, tokenizer


def get_bart_model_and_tokenizer_cnn():
    model = BartForConditionalGeneration.from_pretrained(cnn_model_name)
    tokenizer = BartTokenizer.from_pretrained(cnn_model_name,
                                              force_bos_token_to_be_generated=True,
                                              use_fast=True,
                                              )

    adjust_model(model, tokenizer)
    return model, tokenizer


def get_ranker_model_and_tokenizer(config):
    model = RobertaForSequenceClassification.from_pretrained(ranker_model_name, num_labels=1)
    tokenizer = RobertaTokenizer.from_pretrained(ranker_model_name, use_fast=True)
    adjust_model(model, tokenizer)

    return RankerModel.RankerModel(model, config), tokenizer


def adjust_model(model, tokenizer):
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
