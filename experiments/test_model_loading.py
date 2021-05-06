from config.config import RankerConfig
from experiments import execution_path, experiment

execution_path.set_working_dir()

import utils.compute as compute

torch = compute.get_torch()

from transformers import TrainingArguments
from train import training

from data import generated_data_loading, processing

from unittest import TestCase
import models.model_loading as model_loading


class Test(TestCase):
    def test_get_ranker_model_and_tokenizer(self):
        config = RankerConfig(
            num_examples=4,
            num_skip=2,
            num_summaries_per_text=2,
            learning_rate=1e-5,
            gradient_accumulation_steps=1,
            num_train_epochs=100,
            half_percision=False,
            # half_percision = compute.get_torch().cuda.is_available()
            do_evaluation=False,
            use_dropout=False)
        exp = experiment.start_experiment(hyperparams=config)

        validation_mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum10000_do_sampleFalse_top_pNone_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'

        ranker_model, tokenizer = model_loading.get_ranker_model_and_tokenizer(config)

        validation_generated_xsum = generated_data_loading.load_generated_dataset(validation_mapped_saved_path, 5)
        validation_generated_xsum = validation_generated_xsum.select(
            range(config.num_skip, config.num_skip + config.num_examples))
        validation_processed_generated_xsum = processing.convert_generated_summaries_dataset_to_regression_dataset_format(
            validation_generated_xsum, tokenizer, max_num_summaries_per_text=config.num_summaries_per_text,
            max_seq_len=512)

        print(f'filtered from {len(validation_generated_xsum)} seqs to {len(validation_processed_generated_xsum)}')

        del validation_generated_xsum
        # del validation_processed_generated_xsum
        valid = train = validation_processed_generated_xsum

        training_args = TrainingArguments(
            output_dir="./ranker_output_dir",
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
            evaluation_strategy='epoch' if config.do_evaluation else "no",
            # load_best_model_at_end=True
            dataloader_num_workers=2,
        )

        training.train_ranker(ranker_model, training_args, train,
                              eval_dataset=valid)

# Test().test_get_ranker_model_and_tokenizer()
