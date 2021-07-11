from experiments.gridsearch import gridsearch
from experiments.slurm import run_on_slurm
import random

from models.checkpoints import get_checkpoint_output_dir

model_name = 'facebook/bart-base'
params = {
    'overwrite_output_dir': True,

    'num_train_epochs': '15',
    'evaluation_strategy': 'epoch',
    # 'max_predict_samples': '1000',
    'max_eval_samples': 256,
    'per_device_train_batch_size': '4',
    'per_device_eval_batch_size': '8',
    'do_train': True,
    'do_eval': True,
    'do_predict': True,
    'num_beams': 1,
    'model_name_or_path': model_name,
    'ranker_loss_fn': 'ranking',

    'gradient_accumulation_steps': 2,
    'load_best_model_at_end': True,
    'predict_with_generate': True,
    'save_total_limit': 2,
    # 'greater_is_better': True,
    # 'metric_for_best_model': 'rouge2'
    'metric_for_best_model': 'loss',
    # train each time from scrach cause loading isnt good yet
    'load_generated_model': True,
    'shuffle_training_set': True

    # let it persist the generated datasets one time, for debugging later
    # 'load_generated_model': True,
    # 'shuffle_training_set': False
}

params_for_grid_search = {
    # 'max_train_samples': [8, 16, 32, 64, 128],
    # 'max_train_samples': [16, 24, 32],
    # 'max_train_samples': [8, 16, 32],
    'max_train_samples': [64],
    'shuffle_seed': [32, 10, 12],
    # 'shuffle_seed': [42, 69, 1337],
    # 'max_train_samples': [4, 8, 16, 32],
    # 'max_train_samples': [12, 24],
    # 'max_train_samples': [64],
    'learning_rate': [1e-5],  # todo try 5e-5
    # 'learning_rate': [3e-5],
    'dataset_name': ['xsum'],
    # 'dataset_name': ['cnn_dailymail'],
    # 'dataset_name': ['xsum', 'cnn_dailymail'],

    # 'ranking': ['oracle', 'random'],
    # 'ranking': ['random'],
    # 'ranking': ['oracle'],
    'ranking': ['filter'],
    # 'ranking': ['ensemble'],

    'ranker_loss_fn': ['bce', 'ranking'],
    # 'ranker_loss_fn': ['ranking'],
    # 'ranker_loss_fn': ['bce'],

    'train_filter_on': ['train'],
    # 'train_filter_on': ['train'],
    # 'amount_to_pass_filter': [0.01, 0.05],
    'amount_to_pass_filter': [0.1],
    # 'shuffle_seed': [100, ]
    # 'ranking': ['oracle']
    # 'train_from_scratch_on_unsupervised',
    # 'use_gpt_dataset': [True],
    'train_from_scratch_on_unsupervised': [True]
}

job_name = '''fewshots'''
for p in gridsearch(params, params_for_grid_search):
    dataset_name = p['dataset_name']

    # training_args.shuffle_seed if training_args.shuffle_training_set else None
    p['output_dir'] = get_checkpoint_output_dir(dataset_name, model_name, p["max_train_samples"], p["learning_rate"])
    run_on_slurm(job_name, p, slurm=True)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
