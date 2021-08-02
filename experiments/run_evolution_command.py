from experiments.gridsearch import gridsearch
from experiments.slurm import run_on_slurm
import random

from models.checkpoints import get_checkpoint_output_dir

params_for_grid_search = {
    'skip_first_test_eval': [False],
    'train_filter_on': ['train'],
    # 'dataset_name': ['xsum', 'cnn_dailymail'],
    'dataset_name': ['xsum'],
    'ranking': ['filter'],
    # 'ranking': ['oracle'],
    'ranker_loss_fn': ['ranking'],

    # GPT
    # 'use_gpt_dataset': [True],

    'max_train_samples': [16],
    # 'shuffle_seed': [32, 10, 12],
    'amount_to_pass_filter': [0.01],
    'shuffle_seed': [10],
    # 'ranker_loss_fn': ['ranking', 'bce'],
    # 'ranker_loss_fn': ['ranking', 'bce'],
    'description': 'evolution try',
    'early_stopping_patience': [3],
    'ranker_learning_rate': [3e-5],
    'ranker_gradient_accumulation_steps': [2],
    'ranker_num_epochs': [20],

    # 'max_train_samples': [16, 32, 64],
    'train_from_scratch_on_unsupervised': [False],
    # 'amount_to_pass_filter': [0.01, 0.05],
    # 'ranker_loss_fn': ['ranking', 'bce'],
    # 'max_train_samples': [16, 32, 64],

    # 'amount_to_pass_filter': [0.01, 0.05],
    'use_gpt_dataset': [False],
    'evolution_batch_size': [8, 16]

}
model_name = 'facebook/bart-base'
params = {
    'overwrite_output_dir': True,

    'num_train_epochs': '3',
    'evaluation_strategy': 'epoch',
    # 'max_predict_samples': '1000',
    'max_eval_samples': 32,
    'per_device_train_batch_size': '4',
    'per_device_eval_batch_size': '8',
    'do_train': True,
    'do_eval': True,
    'do_predict': True,
    'num_beams': 1,
    'model_name_or_path': model_name,
    'ranker_loss_fn': 'ranking',
    'ranker_learning_rate': 1e-5,
    'ranker_gradient_accumulation_steps': 2,

    'gradient_accumulation_steps': 2,
    'load_best_model_at_end': True,
    'predict_with_generate': True,
    'save_total_limit': 1,
    # 'greater_is_better': True,
    # 'metric_for_best_model': 'rouge2'
    'metric_for_best_model': 'loss',
    # train each time from scrach cause loading isnt good yet
    'load_generated_model': True,
    'shuffle_training_set': True,
    'learning_rate': 1e-5,  # todo try 5e-5

    # let it persist the generated datasets one time, for debugging later
    # 'load_generated_model': True,
    # 'shuffle_training_set': False
}

job_name = '''evolution'''
for p in gridsearch(params, params_for_grid_search):
    dataset_name = p['dataset_name']

    # training_args.shuffle_seed if training_args.shuffle_training_set else None
    p['output_dir'] = get_checkpoint_output_dir(dataset_name, model_name, p["max_train_samples"], p["learning_rate"])
    run_on_slurm(job_name, p, slurm=True)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
