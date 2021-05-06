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

is_main = False


class Test(TestCase):
    def test_get_ranker_model_and_tokenizer(self):
        config = RankerConfig(
            num_examples=120,
            num_skip=0,
            num_summaries_per_text=4,
            learning_rate=1e-5,
            gradient_accumulation_steps=4,
            num_train_epochs=100,
            half_percision=False,
            # half_percision = compute.get_torch().cuda.is_available()
            do_evaluation=True,
            # evaluate_every_steps=10,
            use_dropout=True,
            print_logits=True)
        exp = experiment.start_experiment(hyperparams=config, tags=['MAIN'] if is_main else None)
        print(config)

        validation_mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum10000_do_sampleFalse_top_pNone_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'

        ranker_model, tokenizer = model_loading.get_ranker_model_and_tokenizer(config)

        validation_processed_generated_xsum = generated_data_loading.load_processed_generated_dataset(
            validation_mapped_saved_path, config, tokenizer)

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
            evaluation_strategy=config.get_evaluation_strategy(),
            # load_best_model_at_end=True
            dataloader_num_workers=2,
            eval_steps=config.evaluate_every_steps
        )

        training.train_ranker(ranker_model, training_args, train,
                              eval_dataset=valid)


if __name__ == '__main__':
    is_main = True
    Test().test_get_ranker_model_and_tokenizer()
