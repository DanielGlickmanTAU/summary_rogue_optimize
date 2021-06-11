from experiments.gridsearch import gridsearch
from experiments.slurm import run_on_slurm

params = {
    'num_examples': 120,
    'num_summaries_per_text': 32,
    'learning_rate': 1e-5,
    'gradient_accumulation_steps': 128,
    'num_train_epochs': 60,
    'loss_fn': 'rank-net',
    'tolerance': 0.08,
    'half_percision': False,
    'do_evaluation': True,
    # 'validation_mapped_saved_path': 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum10000_do_sampleFalse_top_pNone_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0',
    # 'train_mapped_saved_path': 'sshleifer_distilbart-xsum-12-3/processed_dataset__train_xsum50000_do_sampleFalse_top_pNone_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'
    'validation_mapped_saved_path': 'sshleifer_distilbart-cnn-12-3/processed_dataset__validation_xsum1000_do_sampleFalse_top_pNone_top_kNone_num_beams32_num_return_sequences32_no_repeat_ngram_size0',
    'train_mapped_saved_path': 'sshleifer_distilbart-cnn-12-3/processed_dataset__train_xsum1000_do_sampleFalse_top_pNone_top_kNone_num_beams32_num_return_sequences32_no_repeat_ngram_size0',
    # 200k
    # 'train_mapped_saved_path': 'processed_dataset__train_xsum200000_do_sampleFalse_top_pNone_top_kNone_num_beams16_num_return_sequences16_no_repeat_ngram_size0'
}

params_for_grid_search = {

    'learning_rate': [3e-5]

}

import time

job_name = '''run_summerization'''
for p in gridsearch(params, params_for_grid_search):
    run_on_slurm(job_name, p, slurm=True)
