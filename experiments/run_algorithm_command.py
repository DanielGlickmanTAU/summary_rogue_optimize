from experiments.gridsearch import gridsearch
from experiments.slurm import run_on_slurm
import random

from models.checkpoints import get_checkpoint_output_dir

params_for_grid_search = {
    'skip_first_test_eval': [True],
    'train_filter_on': ['train'],
    'ranking': ['filter'],
    'ranker_loss_fn': ['ranking'],

    # GPT
    # 'use_gpt_dataset': [True],
    'dataset_name': ['xsum'],
    # 'ranker_loss_fn': ['ranking', 'bce'],
    # 'ranker_loss_fn': ['ranking', 'bce'],
    # 'ranker_learning_rate': [1e-5, 5e-5],
    # 'train_from_scratch_on_unsupervised': [True, False],

    # self supervisionion, new flow
    #     'use_gpt_dataset': [False],
    #     'use_gpt_dataset': [False],
    #     'dataset_name': ['xsum','cnn_dailymail'],
    #     'ranker_loss_fn': ['ranking', 'bce'],
    #     'ranker_learning_rate': [1e-5, 5e-5],
    #     'amount_to_pass_filter': [0.01, 0.05],

    # 'max_train_samples': [8, 16, 32, 64, 128],
    # 'max_train_samples': [16, 24, 32],
    # 'max_train_samples': [8, 16, 32],
    # 'max_train_samples': [16, 32, 64],
    'train_from_scratch_on_unsupervised': [True, False],
    'amount_to_pass_filter': [0.01, 0.05],
    'ranker_loss_fn': ['ranking', 'bce'],
    'max_train_samples': [16, 32, 64],
    'shuffle_seed': [32, 10, 12],
    # 'amount_to_pass_filter': [0.01, 0.05],
    'use_gpt_dataset': [True],

    # 'shuffle_seed': [12],
    # 'shuffle_seed': [42, 69, 1337],
    # 'max_train_samples': [4, 8, 16, 32],
    # 'max_train_samples': [12, 24],
    # 'max_train_samples': [64],

    # # 'dataset_name': ['xsum'],
    # 'dataset_name': ['xsum', 'cnn_dailymail'],
    # 'ranker_loss_fn': ['ranking'],

    # 'ranking': ['oracle', 'random'],
    # 'ranking': ['oracle'],
    # 'ranking': ['ensemble'],

    # 'ranking': ['random'],
    # 'amount_to_pass_filter': [1.],
    # 'ranker_loss_fn': ['bce'],

    # 'train_filter_on': ['train'],
    # 'amount_to_pass_filter': [0.01, 0.05],

    # 'amount_to_pass_filter': [0.01],
    # 'shuffle_seed': [100, ]
    # 'ranking': ['oracle']
    # 'train_from_scratch_on_unsupervised',
    # 'use_gpt_dataset': [True],

    # ORACLE
    # 'train_from_scratch_on_unsupervised': [True],
    # 'shuffle_seed': [32, 10, 12],
    # # 'max_train_samples': [16],
    # # 'dataset_name': ['xsum'],
    # 'amount_to_pass_filter': [0.01,0.05],
    # 'ranking': ['oracle'],
    # 'use_gpt_dataset': [False],

}
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
    'shuffle_training_set': True,
    'learning_rate': 1e-5,  # todo try 5e-5

    # let it persist the generated datasets one time, for debugging later
    # 'load_generated_model': True,
    # 'shuffle_training_set': False
}

job_name = '''fewshots'''
for p in gridsearch(params, params_for_grid_search):
    dataset_name = p['dataset_name']

    # training_args.shuffle_seed if training_args.shuffle_training_set else None
    p['output_dir'] = get_checkpoint_output_dir(dataset_name, model_name, p["max_train_samples"], p["learning_rate"])
    run_on_slurm(job_name, p, slurm=True)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
