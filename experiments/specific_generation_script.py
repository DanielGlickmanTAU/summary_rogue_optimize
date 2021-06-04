from data import data_loading
from experiments import search_param_setups, experiment
from flows.flows import search_validation_loss
from flows import loading
from models import model_loading
from models.generate import BeamSearchParams

split = 'train'

train_examples = 2_000
validation_examples = 2_000
search_params = BeamSearchParams(num_beams=8, num_return_sequences=8)
batch_size = 2

# exp = experiment.start_experiment(hyperparams={'search': search_params,
#                                                'batch_size': batch_size})
# import websocket
#
# print('websocket version', websocket.__version__)
# 1 / 0

model, tokenizer = model_loading.get_bart_base_model_and_tokenizer()
dataset = data_loading.get_xsum_dataset(train_subset=train_examples, valid_subset=validation_examples)

search_validation_loss(dataset[split], model, tokenizer, search_params, batch_size)
