from models import model_loading, tokenize, generate
from data import cnn_dataset, metrics

batch_size = 16
train_examples = 32

model, tokenizer = model_loading.get_bart_model_and_tokenizer()
cnn = cnn_dataset.get_cnn_dataset(subset=train_examples)
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


dataset = cnn['train'].map(add_summary_and_rouge, batched=True, batch_size=batch_size)

print('rouge1', sum(dataset['rouge2']) / len(dataset['rouge2']))
print('rouge2', sum(dataset['rouge1']) / len(dataset['rouge1']))
