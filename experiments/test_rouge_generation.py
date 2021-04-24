from experiments.flows import search_validation_loss
from flows import loading
from models.generate import PSearchParams

batch_size = 2
train_examples = 50_000
validation_examples = 1
dataset = 'xsum'
# search_params = BeamSearchParams(num_beams=32, num_return_sequences=32)


search_params = PSearchParams(num_beams=8, num_return_sequences=8, top_p=0.9)


def _generate():
    cnn, model, tokenizer = loading.load_dataset_model_tokenizer(dataset, train_examples, validation_examples)
    search_validation_loss(cnn['train'], model, tokenizer, search_params, batch_size)


_generate()
