from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig


def get_bart_model_and_tokenizer():
    # model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6',
                                              force_bos_token_to_be_generated=True
                                              # use_fast=True,
                                              # return_token_type_ids=True
                                              )

    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer
