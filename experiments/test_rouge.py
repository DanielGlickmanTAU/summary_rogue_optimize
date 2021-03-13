from models import model_loading, tokenize, generate
from data import cnn_dataset, metrics

batch_size = 16
train_examples = 16 * 2
validation_examples = 16 * 10

model, tokenizer = model_loading.get_bart_model_and_tokenizer()
cnn = cnn_dataset.get_cnn_dataset(train_subset=train_examples, valid_subset=validation_examples)
rouge = metrics.get_rouge()


def add_summary_and_rouge(examples):
    articles = examples['article']
    gold = examples['highlights']
    generated_summaries = generate.summarize(model, tokenizer, articles)

    assert len(gold) == len(generated_summaries)
    scores = [metrics.calc_score(pred, ref) for pred, ref in zip(generated_summaries, gold)]
    rouge2 = [x['rouge-2'] for x in scores]
    rouge1 = [x['rouge-1'] for x in scores]

    return {'generated_summaries': generated_summaries, 'rouge2': rouge2, 'rouge1': rouge1}


dataset = cnn['validation'].map(add_summary_and_rouge, batched=True, batch_size=batch_size)

print('rouge1 at first 5', sum(dataset['rouge2'][:5]) / 5)
print('rouge2 at first 5', sum(dataset['rouge1'][:5]) / 5)

print('rouge1 at first 16', sum(dataset['rouge2'][:16]) / 16)
print('rouge2 at first 16', sum(dataset['rouge1'][:16]) / 16)

print('rouge1', sum(dataset['rouge2']) / len(dataset['rouge2']))
print('rouge2', sum(dataset['rouge1']) / len(dataset['rouge1']))
