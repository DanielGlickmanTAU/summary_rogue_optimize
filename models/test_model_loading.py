import utils.compute as compute
from data import generated_data_loading

torch = compute.get_torch()
from unittest import TestCase
import models.model_loading as model_loading


class Test(TestCase):
    def test_get_ranker_model_and_tokenizer(self):
        mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum408_do_sampleTrue_top_p0.9_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'
        generated_xsum = generated_data_loading.load_generated_dataset(mapped_saved_path, 4)

        model, tokenizer = model_loading.get_ranker_model_and_tokenizer()
        model(**generated_xsum[0:3])

        # print(encoded_input)
        # output = model(**encoded_input)
        # print(output)
