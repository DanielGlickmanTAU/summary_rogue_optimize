from data import cnn_dataset, metrics
from models import model_loading, generate
from models.candidate_selection import select_best

from train import training

batch_size = 12
train_examples = batch_size * 150
# train_examples = batch_size * 1
validation_examples = batch_size * 50
# validation_examples = batch_size * 1

temperature = 0.5
precentile = 0.15


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
    ds_rouge_ = sum(ds['rouge2']) / len(ds['rouge2'])
    print('rouge2 is ', ds_rouge_, ' evaluate on', len(ds['rouge2']))
    return ds_rouge_


model, tokenizer = model_loading.get_bart_model_and_tokenizer()
cnn = cnn_dataset.get_cnn_dataset(train_subset=train_examples, valid_subset=validation_examples)
rouge = metrics.get_rouge()

strikes = 2

test_summaries = cnn['train'].map(add_summary_and_rouge, batched=True, batch_size=batch_size)
current_valid_score = eval_metric(cnn['validation'])
while True:
    print('selecting top')
    top = select_best(test_summaries, temp=temperature, k=precentile)
    # replace gold tags with generated
    # comment this out when I want to compare to normal train.. and also set select scale_exp=0
    top = top.map(lambda examples: {'highlights': examples['generated_summaries']})
    print('train')
    training.train(model, tokenizer, top, int(batch_size / 2))

    new_valid_score = eval_metric(cnn['validation'])
    if new_valid_score <= current_valid_score:
        strikes = strikes - 1
        if strikes <= 0:
            break

    current_valid_score = new_valid_score
    test_summaries = cnn['train'].map(add_summary_and_rouge, batched=True, batch_size=batch_size)

print('-' * 50)

print('rouge2', sum(test_summaries['rouge2']) / len(test_summaries['rouge2']))
print('rouge1', sum(test_summaries['rouge1']) / len(test_summaries['rouge1']))

print('rouge2 top', sum(top['rouge2']) / len(top['rouge2']))
print('rouge1 top', sum(top['rouge1']) / len(top['rouge1']))
