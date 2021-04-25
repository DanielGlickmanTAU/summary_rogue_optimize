import utils.compute as compute

torch = compute.get_torch()
# from experiments import search_param_setups, execution_path

# execution_path.set_working_dir()

from data import generated_data_loading, processing

from unittest import TestCase
import models.model_loading as model_loading


class Test(TestCase):
    def test_get_ranker_model_and_tokenizer(self):
        mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum408_do_sampleTrue_top_p0.9_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'
        generated_xsum = generated_data_loading.load_generated_dataset(mapped_saved_path, 4)
        generated_xsum = generated_xsum.select(range(10))

        model, tokenizer = model_loading.get_ranker_model_and_tokenizer()
        processed_generated_xsum = processing.convert_generated_summaries_dataset_to_regression_dataset_format(
            generated_xsum, tokenizer)
        model(**processed_generated_xsum[0:3])

        # print(encoded_input)
        # output = model(**encoded_input)
        # print(output)
