from utils import compute

torch = compute.get_torch()
from transformers import BartTokenizer, BartForConditionalGeneration

xsum_model_name = 'sshleifer/distilbart-xsum-12-3'
cnn_model_name = 'sshleifer/distilbart-cnn-12-3'


def get_bart_model_and_tokenizer_xsum():
    model = BartForConditionalGeneration.from_pretrained(xsum_model_name)
    tokenizer = BartTokenizer.from_pretrained(xsum_model_name,
                                              force_bos_token_to_be_generated=True,
                                              use_fast=True,
                                              )

    model.resize_token_embeddings(len(tokenizer))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer


def get_bart_model_and_tokenizer_cnn():
    model = BartForConditionalGeneration.from_pretrained(cnn_model_name)
    tokenizer = BartTokenizer.from_pretrained(cnn_model_name,
                                              force_bos_token_to_be_generated=True,
                                              use_fast=True,
                                              )

    model.resize_token_embeddings(len(tokenizer))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer
