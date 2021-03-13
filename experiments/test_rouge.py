from models import model_loading, tokenize, generate
from data import cnn_dataset, metrics

model, tokenizer = model_loading.get_bart_model_and_tokenizer()
cnn = cnn_dataset.get_cnn_dataset(subset=20)
rouge = metrics.get_rouge()


# articles = cnn['train']['article']
# highlights = cnn['train']['highlights']


def add_summary_and_rouge(examples):
    articles = examples['article']
    gold = examples['highlights']
    generated_summaries = generate.summarize(model, tokenizer, articles)

    assert len(gold) == len(generated_summaries)
    scores = [metrics.calc_score(pred, ref) for pred, ref in zip(generated_summaries, gold)]
    rouge2 = [x['rouge-2'] for x in scores]
    rouge1 = [x['rouge-1'] for x in scores]

    return {'generated_summaries': generated_summaries, 'rouge2': rouge2, 'rouge1': rouge1}


dataset = cnn['train'].map(add_summary_and_rouge, batched=True)

print('rouge1', sum(dataset['rouge2']) / len(dataset['rouge2']))
print('rouge2', sum(dataset['rouge1']) / len(dataset['rouge1']))

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
