import random
import os

from experiments.fewshots.algorithm import rank, filter_dataset
from utils import compute
from experiments.fewshots.learning import do_evaluate, do_train
from config.argument_parsing import parse_generation_args
from data import data_loading, generated_data_loading
from data.processing import convert_dataset_with_generated_highlights_to_training_dataset
from experiments import experiment
from experiments.experiment import log_metrics
from models import model_loading, checkpoints
from models.generate import BeamSearchParams


def load_model_from_checkpoint(model_args, model_checkpoint):
    model_args.model_name_or_path = model_checkpoint
    model, tokenizer = model_loading.get_model_and_tokenizer(model_args)
    return model, tokenizer


def create_and_train_model_on_fewshots(model_args, train_dataset, eval_dataset, training_args, data_args,
                                       last_checkpoint, model_checkpoint):
    model, tokenizer = model_loading.get_model_and_tokenizer(model_args)
    model_save_path = model_checkpoint if training_args.load_generated_model else None
    do_train(model, tokenizer, train_dataset, eval_dataset, training_args, data_args, last_checkpoint,
             model_name_or_path_for_saving=model_save_path)
    return model, tokenizer


def train_model_on_unsupervised_only(model, tokenizer, unsupervised_dataset_for_training, eval_dataset, training_args,
                                     data_args,
                                     model_args, last_checkpoint):
    del (model)
    compute.clean_memory()

    model_args.model_name_or_path = original_model_name_or_path
    model, tokenizer = model_loading.get_model_and_tokenizer(model_args)
    do_train(model, tokenizer, unsupervised_dataset_for_training, eval_dataset, training_args, data_args,
             last_checkpoint, model_name_or_path_for_saving=None)

    return model


data_args, model_args, training_args, last_checkpoint = parse_generation_args()

search_params = BeamSearchParams(num_return_sequences=1, num_beams=data_args.num_beams)

model_checkpoint = \
    checkpoints.get_checkpoint_output_dir(data_args.dataset_name, model_args.model_name_or_path,
                                          data_args.max_train_samples, training_args.learning_rate,
                                          extra=training_args.shuffle_seed if training_args.shuffle_training_set else None)
training_args.output_dir = model_checkpoint + str(random.random())

experiment.start_experiment(hyperparams=[data_args, training_args, model_args],
                            tags=[] if training_args.track_experiment else [
                                'debug'])

original_model_name_or_path = model_args.model_name_or_path

tokenizer = model_loading.get_generator_tokenizer(model_args)

train_dataset, eval_dataset, predict_dataset, unsupervised_data = data_loading.get_dataset(data_args, training_args,
                                                                                           tokenizer,
                                                                                           do_unsupervised=True)

if training_args.load_generated_model and os.path.isdir(model_checkpoint):
    model, tokenizer = load_model_from_checkpoint(model_args, model_checkpoint)
else:
    model, tokenizer = create_and_train_model_on_fewshots(model_args, train_dataset, eval_dataset, training_args,
                                                          data_args,
                                                          last_checkpoint, model_checkpoint)

# eval on test test
rouge_on_test = None

if not training_args.use_gpt_dataset:
    # dont load from cache if not first iteration, because we want to generate again, on new model
    get_cached_generated_summaries = training_args.load_generated_model
    unsupervised_data = generated_data_loading.get_generated_summaries(unsupervised_data, model, tokenizer,
                                                                       search_params,
                                                                       batch_size=training_args.per_device_eval_batch_size,
                                                                       load_generated=get_cached_generated_summaries)

    train_dataset = generated_data_loading.get_generated_summaries(train_dataset, model, tokenizer,
                                                                   search_params,
                                                                   batch_size=training_args.per_device_eval_batch_size,
                                                                   load_generated=get_cached_generated_summaries)

    eval_dataset = generated_data_loading.get_generated_summaries(eval_dataset, model, tokenizer,
                                                                  search_params,
                                                                  batch_size=training_args.per_device_eval_batch_size,
                                                                  load_generated=get_cached_generated_summaries)

ranked_unsupervised_dataset = rank(model, unsupervised_data, train_dataset, eval_dataset, training_args,
                                   search_params)
filtered_unsupervised_dataset = filter_dataset(ranked_unsupervised_dataset, training_args.amount_to_pass_filter)
num_iteration = 1000
batch_size = training_args.evolution_batch_size

for i in range(num_iteration):
    seed_i = batch_size * 1000000 + i + training_args.shuffle_seed * i
    training_subset = filtered_unsupervised_dataset.shuffle(
        seed=seed_i) \
        .select(range(batch_size), keep_in_memory=True)
    unsupervised_dataset_for_training = convert_dataset_with_generated_highlights_to_training_dataset(
        training_subset, tokenizer, data_args)

    model = train_model_on_unsupervised_only(model, tokenizer, unsupervised_dataset_for_training, None,
                                             training_args,
                                             data_args,
                                             model_args, last_checkpoint)
    # clear
    # memory
    # reset
    # model

    final_rouge_on_test = do_evaluate(eval_dataset, model, tokenizer, search_params, description='')
    log_metrics({'rouge2_on_test': final_rouge_on_test})
    train_ids = [training_subset[i]['id'] for i in range(len(training_subset))]
    print(train_ids, final_rouge_on_test)
    with open(f'evo_results_{batch_size}_{i}_{seed_i}', 'w') as f:
        f.write(f'{train_ids}\n{final_rouge_on_test}')
