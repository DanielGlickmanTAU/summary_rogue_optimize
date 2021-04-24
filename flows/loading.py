from data import data_loading
from models import model_loading


def load_dataset_model_tokenizer(dataset_name, train_examples, validation_examples):
    if dataset_name == 'xsum':
        model, tokenizer = model_loading.get_bart_model_and_tokenizer_xsum()
        dataset = data_loading.get_xsum_dataset(train_subset=train_examples, valid_subset=validation_examples)
    else:
        model, tokenizer = model_loading.get_bart_model_and_tokenizer_cnn()
        dataset = data_loading.get_cnn_dataset(train_subset=train_examples, valid_subset=validation_examples)

    return dataset, model, tokenizer
