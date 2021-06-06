import random

import experiment
from data import data_loading
from models.generation import add_summary_and_rouge
from models import model_loading
from models.generate import SearchParams, BeamSearchParams
from train import training
from tqdm import tqdm


def get_random_examples(ds, k):
    indexes = random.sample(range(len(ds)), k)
    return ds.select(indexes)


def eval_metric(model, tokenizer, dataset_split, exp, search_params: SearchParams):
    # compute.clean_memory()

    ds = dataset_split.map(lambda x: add_summary_and_rouge(model, tokenizer, x, search_params),
                           batched=True, batch_size=4)
    print(ds[0]['generated_highlights'])
    ds_rouge_2 = sum(ds['rouge-2-first']) / len(ds['rouge-2-first'])
    ds_rouge_avg = sum(ds['rouge-2-avg']) / len(ds['rouge-2-avg'])
    # ds_rouge_1 = sum(ds['rouge1']) / len(ds['rouge1'])
    print('rouge2 when selecting first beam is ', ds_rouge_2,
          'rouge2 averaging ', search_params.num_beams, ' is ', ds_rouge_avg,
          ' evaluated on', len(ds['rouge-2-first']))
    # print('rouge1 is ', ds_rouge_1, ' evaluate on', len(ds['rouge2']))
    try:
        exp.log_metrics({'rouge2': ds_rouge_2})
    except Exception:
        pass
    return ds_rouge_2


def do_experiment(model, tokenizer, cnn, learning_rate,
                  search_params: SearchParams,
                  gradient_accumulation_steps, batch_size,
                  num_epochs,
                  validation_split='validation'
                  ):
    exp = log_experiment(batch_size, gradient_accumulation_steps, search_params, validation_split)

    train_dataset = cnn['train']
    validation_dataset = cnn[validation_split]
    for i in tqdm(range(num_epochs)):
        training.train(model, tokenizer, train_dataset, validation_dataset, batch_size, learning_rate=learning_rate,
                       gradient_accumulation_steps=gradient_accumulation_steps, num_epochs=10)

        new_valid_score = eval_metric(model, tokenizer, validation_dataset, exp, search_params)
        print(f'rouge 2 on validation in iteration {i} is {new_valid_score}')


def log_experiment(batch_size, gradient_accumulation_steps, search_params, validation_split):
    exp = experiment.start_experiment(hyperparams={
        'batch_size': batch_size,
        'num_beams': search_params.num_beams,
        'num_return_sequences': search_params.num_return_sequences,
        'model_name': model_loading.xsum_model_name,
        'learning_rate': training.learning_rate,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'validation_split': validation_split
    })
    return exp


search_params = BeamSearchParams(num_return_sequences=4, num_beams=4)
model, tokenizer = model_loading.get_bart_base_model_and_tokenizer()
dataset = data_loading.get_xsum_dataset(train_subset=8, valid_subset=8)

validation_split = 'train'
if validation_split != 'validation': print('WARNING TESTING ON ', validation_split)

do_experiment(model, tokenizer, dataset,
              learning_rate=3e-05,
              batch_size=8,
              search_params=search_params,
              gradient_accumulation_steps=1,
              num_epochs=25,
              validation_split=validation_split)
