from experiments.ranker_training_flow import run_exp
from config.config import RankerConfig
from unittest.case import TestCase


class Test(TestCase):
    def test_get_ranker_model_and_tokenizer(self):
        run_exp(config)


config = RankerConfig(
    num_examples=50_000,
    num_skip=0,
    num_summaries_per_text=4,
    learning_rate=1e-5,
    gradient_accumulation_steps=1,
    num_train_epochs=100,
    half_percision=False,
    do_evaluation=True,
    print_logits=True,
    # validation_mapped_saved_path='sshleifer_distilbart-xsum-12-3/processed_dataset__train_xsum12_do_sampleTrue_top_p0.9_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0',
    # train_mapped_saved_path='sshleifer_distilbart-xsum-12-3/processed_dataset__train_xsum12_do_sampleTrue_top_p0.9_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0',
    validation_mapped_saved_path='sshleifer_distilbart-cnn-12-3/processed_dataset__validation_xsum1000_do_sampleFalse_top_pNone_top_kNone_num_beams32_num_return_sequences32_no_repeat_ngram_size0',
    train_mapped_saved_path='sshleifer_distilbart-cnn-12-3/processed_dataset__train_xsum1000_do_sampleFalse_top_pNone_top_kNone_num_beams32_num_return_sequences32_no_repeat_ngram_size0',
    # max_seq_len=999,
    loss_fn='normalized-mse',
    tolerance=0.05
    # evaluate_every_steps=10,

    # 100k
    # validation_mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum10000_do_sampleFalse_top_pNone_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'
    # train_mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__train_xsum50000_do_sampleFalse_top_pNone_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'
)
