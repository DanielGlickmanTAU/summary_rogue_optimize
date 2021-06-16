from experiments.gridsearch import gridsearch
from experiments.slurm import run_on_slurm
import random

model_name = 'facebook/bart-base'
params = {
    'overwrite_output_dir': True,

    'num_train_epochs': '30',
    'evaluation_strategy': 'epoch',
    # 'max_predict_samples': '1000',
    'max_eval_samples': 256,
    'per_device_train_batch_size': '4',
    'per_device_eval_batch_size': '8',
    'do_train': True,
    'do_eval': True,
    'do_predict': True,
    'num_beams': 4,
    'model_name_or_path': model_name,

    'gradient_accumulation_steps': 2,
    'load_best_model_at_end': True,
    'predict_with_generate': True,
    'save_total_limit': 2,
    # 'greater_is_better': True,
    # 'metric_for_best_model': 'rouge2'
    'metric_for_best_model': 'loss'
}

params_for_grid_search = {
    'max_train_samples': [16, 32, 64, 128, 256],
    'learning_rate': [3e-5, 1e-5],
    'dataset_name': ['cnn_dailymail', 'xsum']
}

import time

job_name = '''run_summerization'''
for p in gridsearch(params, params_for_grid_search):
    dataset_name = p['dataset_name']
    p['output_dir'] = f'./models/{dataset_name}/{p["max_train_samples"]}/{model_name}/{p["learning_rate"]}'
    run_on_slurm(job_name, p, slurm=True)
    time.sleep(1)
    print(f'submited {len(p)} jobs')
