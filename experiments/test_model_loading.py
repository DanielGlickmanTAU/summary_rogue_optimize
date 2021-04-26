from experiments import execution_path

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
        mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum408_do_sampleTrue_top_p0.9_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'
        generated_xsum = generated_data_loading.load_generated_dataset(mapped_saved_path, 4)
        generated_xsum = generated_xsum.select(range(20))

        ranker_model, tokenizer = model_loading.get_ranker_model_and_tokenizer()
        processed_generated_xsum = processing.convert_generated_summaries_dataset_to_regression_dataset_format(
            generated_xsum, tokenizer)
        # model(**processed_generated_xsum[0:3])

        training_args = TrainingArguments(
            output_dir="./",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            do_train=True,
            do_eval=False,
            overwrite_output_dir=False,
            # warmup_steps=0,
            fp16=True,
            prediction_loss_only=True,
            learning_rate=1e-5,
            gradient_accumulation_steps=1,
            remove_unused_columns=False,
            # load_best_model_at_end=True
            # dataloader_num_workers=2,
        )

        training.train_ranker(ranker_model, tokenizer, processed_generated_xsum, training_args)

        # print(encoded_input)
        # output = model(**encoded_input)
        # print(output)
