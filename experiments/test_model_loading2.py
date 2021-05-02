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
        # experiment.start_experiment(['score 6k train; 2k valis'])
        ranker_model, tokenizer = model_loading.get_ranker_model_and_tokenizer()
        # mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum408_do_sampleTrue_top_p0.9_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'
        mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__train_xsum50000_do_sampleFalse_top_pNone_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'
        # mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum1200_do_sampleFalse_top_pNone_top_kNone_num_beams32_num_return_sequences32_no_repeat_ngram_size0'
        # validation_mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum1200_do_sampleFalse_top_pNone_top_kNone_num_beams32_num_return_sequences32_no_repeat_ngram_size0'
        validation_mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum10000_do_sampleFalse_top_pNone_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'
        # validation_mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum408_do_sampleTrue_top_p0.9_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'

        # train_generated_xsum = generated_data_loading.load_generated_dataset(mapped_saved_path, 4)
        # train_processed_generated_xsum = processing.convert_generated_summaries_dataset_to_regression_dataset_format(
        #     train_generated_xsum, tokenizer, limit=8)
        # validation_processed_generated_xsum = train_processed_generated_xsum

        num_examples = 5
        num_skip = 0
        num_beams = 6
        learning_rate = 3e-6
        gradient_accumulation_steps = 1
        num_train_epochs = 2500
        # half_percision = compute.get_torch().cuda.is_available()
        half_percision = True
        do_evaluation = True

        validation_generated_xsum = generated_data_loading.load_generated_dataset(validation_mapped_saved_path, 5)
        validation_generated_xsum = validation_generated_xsum.select(range(num_skip, num_skip + num_examples))
        validation_processed_generated_xsum = processing.convert_generated_summaries_dataset_to_regression_dataset_format(
            validation_generated_xsum, tokenizer, limit=num_beams, max_seq_len=512)

        print(f'filtered from {len(validation_generated_xsum)} seqs to {len(validation_processed_generated_xsum)}')

        del validation_generated_xsum
        # del validation_processed_generated_xsum
        valid = train = validation_processed_generated_xsum

        training_args = TrainingArguments(
            output_dir="./ranker_output_dir",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            do_train=True,
            overwrite_output_dir=False,
            # warmup_steps=0,
            fp16=half_percision,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            remove_unused_columns=False,
            evaluation_strategy='epoch' if do_evaluation else "no",
            # load_best_model_at_end=True
            dataloader_num_workers=2,
        )

        # training.train_ranker(ranker_model, tokenizer, training_args, train_processed_generated_xsum,
        #                       eval_dataset=validation_processed_generated_xsum)
        training.train_ranker(ranker_model, training_args, train,
                              eval_dataset=valid)

# Test().test_get_ranker_model_and_tokenizer()
