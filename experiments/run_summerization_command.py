from experiments.gridsearch import gridsearch
from experiments.slurm import run_on_slurm

params = {
    'evaluation_strategy': 'epoch',
    'max_predict_samples': '120',
    'num_train_epochs': '30',
    'per_device_train_batch_size': '4',
    'per_device_eval_batch_size': '8',
    'overwrite_output_dir': True,
    'do_train': True,
    'do_eval': True,
    'max_train_samples': 4,
    'do_predict': True,
    'max_eval_samples': 64,
    'num_beams': 4,
    'model_name_or_path': 'facebook/bart-base',
    'output_dir': './out',
    'dataset_name': 'xsum'
}

params_for_grid_search = {

    'learning_rate': [3e-5]

}

import time

job_name = '''run_summerization'''
for p in gridsearch(params, params_for_grid_search):
    run_on_slurm(job_name, p, slurm=True)
