import random

import experiment
from data import data_loading
from experiments.flows import add_summary_and_rouge, search_validation_loss
from models import model_loading
from models.candidate_selection import select_best
from models.generate import SearchParams, BeamSearchParams, PSearchParams
from train import training


def get_random_examples(ds, k):
    indexes = random.sample(range(len(ds)), k)
    return ds.select(indexes)


def eval_metric(dataset_split, exp, search_params: SearchParams):
    ds = dataset_split.map(lambda x: add_summary_and_rouge(model, tokenizer, x, search_params),
                           batched=True,
                           batch_size=batch_size)
    ds_rouge_2 = sum(ds['rouge2']) / len(ds['rouge2'])
    ds_rouge_1 = sum(ds['rouge1']) / len(ds['rouge1'])
    print('rouge2 is ', ds_rouge_2, ' evaluate on', len(ds['rouge2']))
    try:
        exp.log_metrics({'rouge1': ds_rouge_1, 'rouge2': ds_rouge_2})
    except Exception:
        pass
    return ds_rouge_2


def do_experiment(model, tokenizer, cnn, train_examples, examples_for_training_epoch, learning_rate,
                  search_params: SearchParams, temperature,
                  precentile, gradient_accumulation_steps, strikes=3
                  ):
    exp = experiment.start_experiment(hyperparams={
        'batch_size': batch_size,
        'train_examples': train_examples,
        'validation_examples': validation_examples,
        'temperature': temperature,
        'precentile': precentile,
        'strikes': strikes,
        'top_p': search_params.top_p,
        'top_k': search_params.top_k,
        'do_sample': search_params.do_sample,
        'num_beams': search_params.num_beams,
        'num_return_sequences': search_params.num_return_sequences,
        'model_name': model_loading.xsum_model_name,
        'examples_for_training_batch': examples_for_training_epoch,
        'learning_rate': training.learning_rate,
        'validation_split': validation_split,
        'gradient_accumulation_steps': gradient_accumulation_steps,
    })

    cnn_train = cnn['train']
    test_summaries = get_random_examples(cnn_train, examples_for_training_epoch).map(
        lambda x: add_summary_and_rouge(model, tokenizer, x, search_params),
        batched=True,
        batch_size=batch_size)
    for i in range(10000):
        print('change search_params from 1 to one given in param, possible bug, check this.')
    current_valid_score = eval_metric(cnn[validation_split], exp, search_params)
    while True:
        print('selecting top')
        top = select_best(test_summaries, temp=temperature, k=precentile)
        print('top 3: ', top[:3])
        # replace gold tags with generated
        # comment this out when I want to compare to normal train.. and also set select scale_exp=0
        top = top.map(lambda examples: {'highlights': examples['generated_summaries']})
        print('train')
        training.train(model, tokenizer, top, int(batch_size / 2), learning_rate=learning_rate,
                       gradient_accumulation_steps=gradient_accumulation_steps)

        new_valid_score = eval_metric(cnn[validation_split], exp, search_params)
        if new_valid_score <= current_valid_score:
            strikes = strikes - 1
            if strikes <= 0:
                break

        current_valid_score = new_valid_score
        test_summaries = get_random_examples(cnn_train, examples_for_training_epoch).map(
            lambda x: add_summary_and_rouge(model, tokenizer, x, search_params),
            batched=True,
            batch_size=batch_size)
    print('done single expirment')
    exp.end()


validation_split = 'validation'

batch_size = 12
train_examples = 408
validation_examples = 408
examples_for_training_epoch = 3200
examples_for_training_epoch = 16
strikes = 3
temperature = 2.5
precentile = 0.06

model, tokenizer = model_loading.get_bart_model_and_tokenizer_xsum()
cnn = data_loading.get_xsum_dataset(train_subset=train_examples, valid_subset=validation_examples)

search_params = BeamSearchParams(num_beams=32, num_return_sequences=32)
search_validation_loss(cnn['validation'], model, tokenizer, search_params, batch_size)

search_params = PSearchParams(num_beams=32, num_return_sequences=32, top_p=0.9)
search_validation_loss(cnn['validation'], model, tokenizer, search_params, batch_size)

# search_params = BeamSearchParams(num_beams=16, num_return_sequences=16, no_repeat_ngram_size=2)
# search_validation_loss(cnn['train'], model, tokenizer, search_params, batch_size)

exit();
1 / 0

search_params = SearchParams(do_sample=False, top_p=None, top_k=100, num_beams=4, num_return_sequences=4)
do_experiment(model, tokenizer, cnn,
              train_examples=4000,
              examples_for_training_epoch=100,
              learning_rate=1e-05,
              temperature=10,
              precentile=0.1,
              search_params=search_params,
              strikes=10,
              gradient_accumulation_steps=2,
              )
