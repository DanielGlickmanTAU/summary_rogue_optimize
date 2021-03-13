from models import model_loading, tokenize
from data import cnn_dataset, metrics
import torch

model, tokenizer = model_loading.get_bart_model_and_tokenizer()
cnn = cnn_dataset.get_cnn_dataset(subset=1)
rouge = metrics.get_rouge()


# articles = cnn['train']['article']
# highlights = cnn['train']['highlights']


def process(examples):
    articles = cnn['train']['article']
    highlights = cnn['train']['highlights']

    inputs = tokenize.tokenize(tokenizer, articles)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=1024)
    decoded_summary_generated = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g
                                 in
                                 summary_ids]
# print(decoded_summary_generated)
#
# fixed = cnn['train'] \
#     .map(
#     lambda examples: tokenizer(examples['article'], max_length=1024, padding=True,
#                                truncation=True), batched=True).map(
#     lambda examples: {
#         'summary_ids': model.generate(torch.tensor(examples['input_ids']), num_beams=4, max_length=1024).numpy()},
#     batched=True).map(
#     lambda examples: {
#         'summary_string': [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
#                            examples['summary_ids']]}, batched=True)
#
# print(fixed)
