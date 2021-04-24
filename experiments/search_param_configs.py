from data import data_loading
from models import model_loading
from models.generate import BeamSearchParams


def get_xsum_beam_train_config():
    batch_size = 2
    train_examples = 50_000
    validation_examples = 1
    search_params = BeamSearchParams(num_beams=32, num_return_sequences=32)

    dataset_name = 'xsum'
    if dataset_name == 'xsum':
        model, tokenizer = model_loading.get_bart_model_and_tokenizer_xsum()
        dataset = data_loading.get_xsum_dataset(train_subset=train_examples, valid_subset=validation_examples)
    else:
        model, tokenizer = model_loading.get_bart_model_and_tokenizer_cnn()
        dataset = data_loading.get_cnn_dataset(train_subset=train_examples, valid_subset=validation_examples)
