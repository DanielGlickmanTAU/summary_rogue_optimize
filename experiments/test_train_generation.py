import random

import experiment
from data import data_loading
from models.generation import add_summary_and_rouge
from models import model_loading
from models.candidate_selection import select_best
from models.generate import SearchParams, BeamSearchParams
from train import training


def get_random_examples(ds, k):
    indexes = random.sample(range(len(ds)), k)
    return ds.select(indexes)


def eval_metric(dataset_split, exp, search_params: SearchParams):
    ds = dataset_split.map(lambda x: add_summary_and_rouge(model, tokenizer, x, search_params),
                           batched=True)
    ds_rouge_2 = sum(ds['rouge2']) / len(ds['rouge2'])
    ds_rouge_1 = sum(ds['rouge1']) / len(ds['rouge1'])
    print('rouge2 is ', ds_rouge_2, ' evaluate on', len(ds['rouge2']))
    print('rouge1 is ', ds_rouge_1, ' evaluate on', len(ds['rouge2']))
    try:
        exp.log_metrics({'rouge1': ds_rouge_1, 'rouge2': ds_rouge_2})
    except Exception:
        pass
    return ds_rouge_2


def do_experiment(model, tokenizer, cnn, learning_rate,
                  search_params: SearchParams,
                  gradient_accumulation_steps, batch_size,
                  num_epochs,
                  validation_split='validation'
                  ):
    exp = experiment.start_experiment(hyperparams={
        'batch_size': batch_size,
        'top_p': search_params.top_p,
        'top_k': search_params.top_k,
        'do_sample': search_params.do_sample,
        'num_beams': search_params.num_beams,
        'num_return_sequences': search_params.num_return_sequences,
        'model_name': model_loading.xsum_model_name,
        'learning_rate': training.learning_rate,
        'validation_split': validation_split,
        'gradient_accumulation_steps': gradient_accumulation_steps,
    })

    cnn_train = cnn['train']

    print('train')
    training.train(model, tokenizer, cnn_train, batch_size, learning_rate=learning_rate,
                   gradient_accumulation_steps=gradient_accumulation_steps, num_epochs=num_epochs)

    new_valid_score = eval_metric(cnn[validation_split], exp, search_params)
    print(new_valid_score)


search_params = BeamSearchParams(num_return_sequences=4, num_beams=4)
model, tokenizer = model_loading.get_bart_base_model_and_tokenizer()
dataset = data_loading.get_xsum_dataset(train_subset=1_000, valid_subset=1_000)

do_experiment(model, tokenizer, dataset,
              learning_rate=3e-05,
              batch_size=8,
              search_params=search_params,
              gradient_accumulation_steps=10,
              num_epochs=40)