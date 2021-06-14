from experiments.gridsearch import gridsearch
from experiments.slurm import run_on_slurm
import random

dataset_name = 'xsum'
model_name = 'facebook/bart-base'
train_samples = 100
params = {
    'num_train_epochs': '30',
    'evaluation_strategy': 'epoch',
    'max_predict_samples': '1000',
    'per_device_train_batch_size': '4',
    'per_device_eval_batch_size': '8',
    'overwrite_output_dir': False,
    'do_train': True,
    'do_eval': True,
    'do_predict': True,
    'max_eval_samples': 128,
    'num_beams': 4,
    'model_name_or_path': model_name,

    'dataset_name': dataset_name,
    'gradient_accumulation_steps': 2,
    'load_best_model_at_end': True,
    'predict_with_generate': True,
    'save_total_limit': 1,
    'greater_is_better': True,
    'metric_for_best_model': 'rouge2'
}

params_for_grid_search = {
    'max_train_samples': [16, 32, 64, 128, 256],
    # 'learning_rate': [1e-5, 3e-5, 10e-5]
    'learning_rate': [1e-5]

}

import time

job_name = '''run_summerization'''
for p in gridsearch(params, params_for_grid_search):
    p['output_dir'] = f'./models/{dataset_name}/{p["max_train_samples"]}/{model_name}/{p["learning_rate"]}'
    run_on_slurm(job_name, p, slurm=True)
    time.sleep(1)
