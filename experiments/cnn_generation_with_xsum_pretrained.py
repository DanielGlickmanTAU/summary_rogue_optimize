from data import data_loading
from experiments import search_param_setups
from flows.flows import search_validation_loss
from flows import loading
from models import model_loading

dataset_name, split, train_examples, validation_examples, search_params, batch_size = \
    search_param_setups.get_cnn_beam_validation_setup()

assert dataset_name == 'cnn'

split = 'validation'
train_examples = 1
validation_examples = 1000

# split = 'train'
# train_examples = 1000
# validation_examples = 1

model, tokenizer = model_loading.get_bart_model_and_tokenizer_cnn()
dataset = data_loading.get_xsum_dataset(train_subset=train_examples, valid_subset=validation_examples)

search_validation_loss(dataset[split], model, tokenizer, search_params, batch_size)
