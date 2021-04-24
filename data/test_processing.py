from experiments import search_param_setups, execution_path
from models import model_loading

execution_path.set_working_dir()
from unittest import TestCase
import processing

from data import generated_data_loading
from experiments import execution_path
from flows import loading
from models.generate import PSearchParams
import os


class Test(TestCase):
    def test_convert_generated_summaries_dataset_to_regression_dataset_format(self):
        # dataset_name, split, train_examples, validation_examples, search_params, batch_size = search_param_setups.get_xsum_spread_search_setup()
        # generated_data_loading.get_generated_dataset_save_path()
        #
        # search_params = PSearchParams(num_beams=8, num_return_sequences=8, top_p=0.9)
        model, tokenizer = model_loading.get_bart_model_and_tokenizer_xsum()
        mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum1200_do_sampleFalse_top_pNone_top_kNone_num_beams32_num_return_sequences32_no_repeat_ngram_size0'
        generated_xsum = generated_data_loading.load_generated_dataset(mapped_saved_path, 4)
        generated_xsum = generated_xsum.select(range(10))
        first_text = generated_xsum['article'][0]
        processed_generated_xsum = processing.convert_generated_summaries_dataset_to_regression_dataset_format(
            generated_xsum, tokenizer)

        first_highlight = generated_xsum['generated_highlights'][0][0]
        custom_input = processed_generated_xsum['custom_input'][0][0]
        decoded_input = tokenizer.decode(custom_input)
        assert first_text in decoded_input
        assert first_highlight in decoded_input
        assert tokenizer.sep_token in decoded_input
