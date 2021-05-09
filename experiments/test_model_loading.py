import utils.compute as compute
from config.config import RankerConfig
from experiments import execution_path, experiment
from config import argument_parsing

execution_path.set_working_dir()

torch = compute.get_torch()

from transformers import TrainingArguments
from train import training

from data import generated_data_loading

from unittest import TestCase
import models.model_loading as model_loading
import time


class Test(TestCase):
    def test_get_ranker_model_and_tokenizer(self):
        config = RankerConfig(
            num_examples=50_000,
            num_skip=0,
            num_summaries_per_text=4,
            learning_rate=1e-5,
            gradient_accumulation_steps=16,
            num_train_epochs=10,
            half_percision=False,
            do_evaluation=True,
            validation_mapped_saved_path='sshleifer_distilbart-xsum-12-3/processed_dataset__train_xsum12_do_sampleTrue_top_p0.9_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0',
            train_mapped_saved_path='sshleifer_distilbart-xsum-12-3/processed_dataset__train_xsum12_do_sampleTrue_top_p0.9_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0',
            # max_seq_len=400

            # evaluate_every_steps=10,
            # validation_mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum10000_do_sampleFalse_top_pNone_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'
            # train_mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__train_xsum50000_do_sampleFalse_top_pNone_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'
        )

        run_exp(config)


def run_exp(config):
    tags = [f'num train examples{config.num_examples}', f'summaries per text{config.num_summaries_per_text}',
            config.train_mapped_saved_path,
            config.validation_mapped_saved_path,
            config.loss_fn]
    exp = experiment.start_experiment(hyperparams=vars(config), tags=tags)
    print(config)
    ranker_model, tokenizer = model_loading.get_ranker_model_and_tokenizer(config)
    validation_processed_generated_xsum = generated_data_loading.load_processed_generated_dataset(
        config.validation_mapped_saved_path, config, tokenizer)
    train_processed_generated_xsum = generated_data_loading.load_processed_generated_dataset(
        config.train_mapped_saved_path, config, tokenizer)
    training_args = TrainingArguments(
        output_dir="./ranker_output_dir_" + str(time.time()).replace('.', '_'),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        do_train=True,
        overwrite_output_dir=False,
        # warmup_steps=0,
        fp16=config.half_percision,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        remove_unused_columns=False,
        evaluation_strategy='steps' if config.evaluate_every_steps else 'epoch' if config.do_evaluation else 'no',
        # load_best_model_at_end=True
        dataloader_num_workers=2,
        eval_steps=config.evaluate_every_steps,
        report_to=["comet_ml"],
        # load_best_model_at_end=load_best_model_at_end,
        # metric_for_best_model=metric_name,
        save_total_limit=1
    )
    training.train_ranker(ranker_model, config,
                          training_args, train_processed_generated_xsum,
                          eval_dataset=validation_processed_generated_xsum)


config = argument_parsing.get_args()
run_exp(config)
