import os
import sys
import time

from experiments.gridsearch import gridsearch

print(os.path)
python = os.sys.executable

slurm_file = 'my_slurm.slurm'


# job_name = '''argument_parsing'''
# job_name = '''slurm_test'''


def run_on_slurm(job_name, params, slurm=True, gpu=True):
    python_file = job_name
    python_file = python_file.replace('.py', '')
    job_name = job_name + str(time.time())
    if slurm:
        with open(slurm_file, 'w') as f:
            f.write(f'''#! /bin/sh
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err
#SBATCH -p {partition}
## SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus={'1' if gpu else '0'}
{python} {python_file}.py ''' + ' '.join([f'--{key} {value}' for key, value in params.items()]))

        print(f'executing {job_name} ')
        os.system(f'sbatch {slurm_file}')
    else:
        f = f'{python} {python_file}.py ' + ' '.join([f'--{key} {value}' for key, value in params.items()])
        os.system(f"nohup sh -c ' {f} > res.txt '&")
    os.system('rm slurm.py')


# job_name = 'cnn_generation_with_xsum_pretrained'

# partition, time_limit = 'studentbatch', '3-00:00:00'

partition, time_limit = 'studentkillable', 'infinite'

# partition, time_limit = 'studentrun', '33:00:00'

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
    'num_summaries_per_text': [32, 16, 8],
    'learning_rate': [3e-5, 2e-5, 1e-5],
    'gradient_accumulation_steps': [32, 16, 8, 4, 2],
    'loss_fn': ['normalized-mse', 'centered-mse', 'rank-net'],
    # 'loss_fn': ['ranking'],
    # 'tolerance': [0.04, 0.08, 0.1],
}

import time

if __name__ == '__main__':
    # job_name = '''test_model_loading'''
    # for p in gridsearch(params, params_for_grid_search):
    #     run_on_slurm(job_name, p, slurm=True)
    #     time.sleep(120)

    # job_name = '''specific_generation_script'''
    job_name = '''test_rouge_generation'''
    run_on_slurm(job_name, {}, slurm=True)
