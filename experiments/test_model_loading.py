import os

from experiments.ranker_training_flow import run_exp

for key in list(os.environ.keys()):
    os.environ[key] = os.environ[key].replace('chaimc', 'glickman1')
import utils.compute as compute
from experiments import execution_path
from config import argument_parsing

execution_path.set_working_dir()

torch = compute.get_torch()

config = argument_parsing.get_ranker_config()
run_exp(config)
