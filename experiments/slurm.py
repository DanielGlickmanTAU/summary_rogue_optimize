import os
import sys

slurm_file = 'my_slurm.slurm'
# partition = 'studentbatch'
# partition = 'studentbatch'
partition = 'studentrun'
with open(slurm_file, 'w') as f:
    f.write('''#! /bin/sh
#SBATCH --job-name=awesome
#SBATCH --output=awesome.out
#SBATCH --error=awesome.err
#SBATCH --partition=%s
#SBATCH --job-name hello-world
#SBATCH --time=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
python slurm_test.py''' % partition)

# os.system("nohup sh -c '" +
#           sys.executable + " slu > res.txt " +
#           "' &")


os.system(f'sbatch {slurm_file}')
