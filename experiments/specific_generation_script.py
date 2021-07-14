from data import data_loading
from experiments import search_param_setups, experiment
from flows.flows import search_validation_loss
from models import model_loading
from models.generate import BeamSearchParams

split = 'train'

train_examples = 200
validation_examples = 128
test_examples = 1000
search_params = BeamSearchParams(num_beams=4, num_return_sequences=4)
batch_size = 2

# exp = experiment.start_experiment(hyperparams={'search': search_params,
#                                                'batch_size': batch_size})

checkpoint_ = 'models/xsum/100/facebook/bart-base/1e-05'
model, tokenizer = model_loading._get_bart_based_model_and_tokenizer(checkpoint_)
# model, tokenizer = model_loading.get_bart_base_model_and_tokenizer()

dataset = data_loading.get_xsum_dataset(train_subset=train_examples, valid_subset=validation_examples,
                                        test_subset=test_examples)
#
search_validation_loss(dataset[split], model, tokenizer, search_params, batch_size)
