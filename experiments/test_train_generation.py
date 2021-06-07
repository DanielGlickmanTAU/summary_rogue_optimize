import random

import experiment
from data import data_loading
from models import model_loading
from models.generate import SearchParams, BeamSearchParams
from train import training
from train.generation_training import generation_train_flow


def get_random_examples(ds, k):
    indexes = random.sample(range(len(ds)), k)
    return ds.select(indexes)


def do_experiment(model, tokenizer, cnn, learning_rate,
                  search_params: SearchParams,
                  gradient_accumulation_steps, batch_size,
                  num_epochs,
                  validation_split='validation'
                  ):
    exp = log_experiment(batch_size, gradient_accumulation_steps, search_params, validation_split)

    train_dataset = cnn['train']
    validation_dataset = cnn[validation_split]

    generation_train_flow(model, tokenizer, exp, search_params, train_dataset, validation_dataset,
                          batch_size,
                          learning_rate, gradient_accumulation_steps, num_epochs)


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
