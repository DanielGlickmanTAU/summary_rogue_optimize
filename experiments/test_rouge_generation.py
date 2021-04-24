from experiments.flows import search_validation_loss
from data import data_loading
from models import model_loading
from models.generate import PSearchParams, BeamSearchParams

batch_size = 2
train_examples = 1
validation_examples = 1200
dataset = 'xsum'
search_params = BeamSearchParams(num_beams=32, num_return_sequences=32)
search_params = PSearchParams(num_beams=64, num_return_sequences=64, top_p=0.9)


def _generate():
    if dataset == 'xsum':
        model, tokenizer = model_loading.get_bart_model_and_tokenizer_xsum()
        cnn = data_loading.get_xsum_dataset(train_subset=train_examples, valid_subset=validation_examples)
    else:
        model, tokenizer = model_loading.get_bart_model_and_tokenizer_cnn()
        cnn = data_loading.get_cnn_dataset(train_subset=train_examples, valid_subset=validation_examples)
    search_validation_loss(cnn['validation'], model, tokenizer, search_params, batch_size)


_generate()
