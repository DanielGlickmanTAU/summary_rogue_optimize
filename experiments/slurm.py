import os
import sys
import time

print(os.path)
python = os.sys.executable
# path =
# 1 / 0
slurm_file = 'my_slurm.slurm'
# job_name = '''test_model_loading'''
job_name = '''tmptmp'''
# job_name = '''slurm_test'''

# partition = 'studentrun'
partition, time_limit = 'studentbatch', '3-00:00:00'
# partition, time_limit = 'studentkillable', 'infinite'

params = {
    'do_evaluation': False,
    'gradient_accumulation_steps': 4123
}

python_file = job_name
python_file = python_file.replace('.py', '')
job_name = job_name + str(time.time())
with open(slurm_file, 'w') as f:
    f.write(f'''#! /bin/sh
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err
#SBATCH -p {partition}
## SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
{python} {python_file}.py ''' + ' '.join([f'--{key} {value}' for key, value in params.items()]))
## SBATCH --time=infinite

print(f'executing {job_name} ')
os.system(f'sbatch {slurm_file}')
