import experiment
from data import data_loading, metrics
from models import model_loading, generate
from models.candidate_selection import select_best
from train import training

batch_size = 6
train_examples = 2500
# train_examples = batch_size * 1
validation_examples = 600
# validation_examples = batch_size * 1

strikes = 3
temperature = 0.6
precentile = 0.1

exp = experiment.start_experiment(hyperparams={
    'batch_size': batch_size,
    'train_examples': train_examples,
    'validation_examples': validation_examples,
    'temperature': temperature,
    'precentile': precentile,
    'strikes': strikes,
    'top_p': generate.top_p,
    'top_k': generate.top_k,
    'do_sample': generate.do_sample,
    'num_beams': generate.num_beams,
    'num_return_sequences': generate.num_return_sequences,
    'model_name': model_loading.model_name
})


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
    ds = dataset_split.map(add_summary_and_rouge, batched=True, batch_size=batch_size)
    ds_rouge_2 = sum(ds['rouge2']) / len(ds['rouge2'])
    ds_rouge_1 = sum(ds['rouge1']) / len(ds['rouge1'])
    print('rouge2 is ', ds_rouge_2, ' evaluate on', len(ds['rouge2']))
    try:
        exp.log_metrics({'rouge1': ds_rouge_1, 'rouge2': ds_rouge_2})
    except Exception:
        pass
    return ds_rouge_2


model, tokenizer = model_loading.get_bart_model_and_tokenizer()
cnn = data_loading.get_cnn_dataset(train_subset=train_examples, valid_subset=validation_examples)
rouge = metrics.get_rouge()

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
