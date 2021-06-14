from experiments import search_param_setups, execution_path

execution_path.set_working_dir()
from models import model_loading

from unittest import TestCase
import processing

from data import generated_data_loading


class Test(TestCase):
    def test_convert_generated_summaries_dataset_to_regression_dataset_format(self):
        model, tokenizer = model_loading.get_bart_model_and_tokenizer_xsum()
        # mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum1200_do_sampleFalse_top_pNone_top_kNone_num_beams32_num_return_sequences32_no_repeat_ngram_size0'
        mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum408_do_sampleTrue_top_p0.9_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'
        # mapped_saved_path = 'processed_dataset__train_xsum50000_do_sampleFalse_top_pNone_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'
        generated_xsum = generated_data_loading.load_generated_dataset(mapped_saved_path, 4)
        # generated_xsum = generated_xsum.select(range(10))
        processed_generated_xsum = processing.convert_generated_summaries_dataset_to_regression_dataset_format(
            generated_xsum, tokenizer)
        first_text = generated_xsum['article'][0]
        first_highlight = generated_xsum['generated_highlights'][0][0]
        custom_input = processed_generated_xsum['input_ids_s'][0][0]

        decoded_input = tokenizer.decode(custom_input)
        assert first_text in decoded_input
        assert first_highlight in decoded_input
        assert tokenizer.sep_token in decoded_input
