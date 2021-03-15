import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

#model_name = 'facebook/bart-large'
# model_name = 'sshleifer/distilbart-6-6-cnn'
model_name = 'sshleifer/distilbart-cnn-12-6'


def get_bart_model_and_tokenizer():
    # model = BartForConditionalGeneration.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    tokenizer = BartTokenizer.from_pretrained(model_name,
                                              force_bos_token_to_be_generated=True,
                                              use_fast=True,
                                              )

    model.resize_token_embeddings(len(tokenizer))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer
