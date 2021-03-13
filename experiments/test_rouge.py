from data import cnn_dataset, metrics
from models import model_loading, generate
from models.candidate_selection import select_best

batch_size = 16
train_examples = 16 * 2
validation_examples = 16 * 2

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


def eval_metric(dataset_split):
    ds = dataset_split.map(add_summary_and_rouge, batched=True, batch_size=batch_size, keep_in_memory=True)
    return sum(ds['rouge2']) / len(ds['rouge2'])


def train(model, tokenizer, data):
    pass

test_summaries = cnn['test'].map(add_summary_and_rouge, batched=True, batch_size=batch_size)
valid = eval_metric(cnn['validation'])
while True:
    top = select_best(test_summaries)
    # replace gold tags with generated
    # comment this out when I want to compare to normal training
    top = top.map(lambda examples: {'highlights': examples['generated_summaries']})
    train(model, tokenizer, top)

print('rouge2', sum(test_summaries['rouge2']) / len(test_summaries['rouge2']))
print('rouge1', sum(test_summaries['rouge1']) / len(test_summaries['rouge1']))

print('rouge2 top', sum(top['rouge2']) / len(top['rouge2']))
print('rouge1 tםפ', sum(top['rouge1']) / len(top['rouge1']))
