from config.argument_parsing import parse_generation_args
from data import data_loading, generated_data_loading
from data.processing import convert_dataset_with_generated_highlights_to_training_dataset
from evaluation import evaluate
from experiments import experiment
from experiments.experiment import log_metrics
from models import model_loading, generation, checkpoints
from time import time
import random

# TODO this also exists in run_summarization, origanize and move this to one place
from models.generate import BeamSearchParams
from train import generation_training
from utils import decorators


@decorators.measure_time
def my_eval(dataset, model, tokenizer, search_params, description=''):
    ds = generation.add_summary_and_rouge(model, tokenizer, dataset,
                                          search_params)
    print(f'evaluate {description}')
    return evaluate.print_rouge_stuff(ds)


@decorators.measure_time
def do_train(model, tokenizer, train_dataset, eval_dataset, training_args, data_args, last_checkpoint):
    # need this because the trainer remove features that are not neccessery for the model(like article and highlights), which messes things up later.
    train_dataset = train_dataset.map()
    eval_dataset = eval_dataset.map()
    trainer = generation_training.create_trainer(train_dataset, eval_dataset, training_args, data_args, model,
                                                 tokenizer)

    checkpoint = last_checkpoint
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    # elif last_checkpoint is not None:
    #     checkpoint = last_checkpoint
    if checkpoint:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
    else:
        train_result = trainer.train()

    if training_args.save_model_after_train:
        t = time()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()
        print(f'saving took {time() - t} seconds')
    else:
        print('skiping saving generation model')
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    return metrics


data_args, model_args, training_args, last_checkpoint = parse_generation_args()
search_params = BeamSearchParams(num_return_sequences=1, num_beams=data_args.num_beams)

model_checkpoint = \
    checkpoints.get_checkpoint_output_dir(data_args.dataset_name, model_args.model_name_or_path,
                                          data_args.max_train_samples, training_args.learning_rate, extra=None)
training_args.output_dir = model_checkpoint + str(random.random())

experiment.start_experiment(hyperparams=[data_args, training_args, model_args],
                            tags=[] if training_args.track_experiment else ['debug'])

if training_args.load_generated_model:
    if training_args.shuffle_training_set:
        raise NotImplementedError('it is not supported right now with get_generated_summaries')
    model_args.model_name_or_path = model_checkpoint
    model, tokenizer = model_loading.get_model_and_tokenizer(model_args)
else:
    model, tokenizer = model_loading.get_model_and_tokenizer(model_args)

train_dataset, eval_dataset, predict_dataset, unsupervised_data = data_loading.get_dataset(data_args, training_args,
                                                                                           tokenizer,
                                                                                           do_unsupervised=True)

if not training_args.load_generated_model:
    do_train(model, tokenizer, train_dataset, eval_dataset, training_args, data_args, last_checkpoint)

my_eval(train_dataset, model, tokenizer, search_params,
        f'on TRAIN set after training on {len(train_dataset)} samples')

my_eval(eval_dataset, model, tokenizer, search_params,
        f'on eval set after training on {len(train_dataset)} samples')

unsupervised_data = generated_data_loading.get_generated_summaries(unsupervised_data, model, tokenizer,
                                                                   search_params,
                                                                   batch_size=training_args.per_device_eval_batch_size,
                                                                   load_generated=training_args.load_generated_model)


def rank(unsupervised_data, ranking):
    if ranking == 'oracle':
        unsupervised_data_with_rouge = generated_data_loading.get_generated_rouge(unsupervised_data, model,
                                                                                  search_params,
                                                                                  training_args.load_generated_model)
        return unsupervised_data_with_rouge.map(lambda example: {'rank': example['rouge-2-first']})
    if ranking == 'random':
        return unsupervised_data.map(lambda example: {'rank': random.random()})
    raise Exception('unknown ranking', ranking)


def filter(ranked_dataset, amount_to_pass_filter=0.01):
    ranked_dataset = ranked_dataset.sort('rank', reverse=True)
    return ranked_dataset.select(range(max(1, int(amount_to_pass_filter * len(ranked_dataset)))))


ranked_unsupervised_dataset = rank(unsupervised_data, training_args.ranking)
filtered_unsupervised_dataset = filter(ranked_unsupervised_dataset, training_args.amount_to_pass_filter)
unsupervised_dataset_for_training = convert_dataset_with_generated_highlights_to_training_dataset(
    filtered_unsupervised_dataset, tokenizer, data_args)

do_train(model, tokenizer, unsupervised_dataset_for_training, eval_dataset, training_args, data_args, last_checkpoint)
my_eval(eval_dataset, model, tokenizer, search_params, description='on eval set after training unsupervised')

final_rouge_on_test = my_eval(predict_dataset, model, tokenizer, search_params, description='on test set now')
log_metrics({'rouge2_on_test': final_rouge_on_test})
# generator model , generator tokenizer =


# train generator(train,validation)

# create summries and add rouge for train,validation, test

# train filter(train, validation)


# rank  unsupervised generated

# select top ranked

# insert into train set(can use some dataset feature to replace 'generated_highlight' for creating labels for generation trainig)
