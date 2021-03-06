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


def train_model_on_unsupervised(model, tokenizer, unsupervised_dataset_for_training, eval_dataset, training_args,
                                data_args,
                                model_args, last_checkpoint):
    if training_args.train_from_scratch_on_unsupervised:
        del (model)
        compute.clean_memory()

        model_args.model_name_or_path = original_model_name_or_path
        model, tokenizer = model_loading.get_model_and_tokenizer(model_args)
        do_train(model, tokenizer, unsupervised_dataset_for_training, eval_dataset, training_args, data_args,
                 last_checkpoint, model_name_or_path_for_saving=None)
        do_train(model, tokenizer, train_dataset, eval_dataset, training_args, data_args, last_checkpoint,
                 model_name_or_path_for_saving=None)
    else:
        do_train(model, tokenizer, unsupervised_dataset_for_training, eval_dataset, training_args, data_args,
                 last_checkpoint, model_name_or_path_for_saving=None)
    return model


data_args, model_args, training_args, last_checkpoint = parse_generation_args()
training_args.load_generated_model = False

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

train_dataset, eval_dataset, predict_dataset = data_loading.get_dataset(data_args, training_args,
                                                                        tokenizer,
                                                                        do_unsupervised=False)

train_dataset = train_dataset.shuffle(training_args.training_set_shuffle_seed)
model, tokenizer = create_and_train_model_on_fewshots(model_args, train_dataset, eval_dataset, training_args,
                                                      data_args,
                                                      last_checkpoint, model_checkpoint)

rouge_on_test = do_evaluate(predict_dataset, model, tokenizer, search_params,
                            f'on TEST set after training on {len(train_dataset)} samples')
log_metrics({'rouge2_on_test': rouge_on_test})
