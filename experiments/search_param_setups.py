from models.generate import BeamSearchParams, PSearchParams


def get_xsum_beam_train_setup():
    dataset_name = 'xsum'
    split = 'train'
    train_examples = 50_000
    validation_examples = 1
    search_params = BeamSearchParams(num_beams=8, num_return_sequences=8)
    batch_size = 2
    return dataset_name, split, train_examples, validation_examples, search_params, batch_size


def get_cnn_beam_train_setup():
    dataset_name = 'cnn'
    split = 'train'
    train_examples = 20_000
    validation_examples = 1
    search_params = BeamSearchParams(num_beams=32, num_return_sequences=32)
    batch_size = 2
    return dataset_name, split, train_examples, validation_examples, search_params, batch_size


def get_cnn_beam_validation_setup():
    dataset_name = 'cnn'
    split = 'validation'
    train_examples = 1
    validation_examples = 10_000
    search_params = BeamSearchParams(num_beams=32, num_return_sequences=32)
    batch_size = 2
    return dataset_name, split, train_examples, validation_examples, search_params, batch_size


def get_xsum_beam_train_FULL_setup():
    dataset_name = 'xsum'
    split = 'train'
    train_examples = 200_000
    validation_examples = 1
    search_params = BeamSearchParams(num_beams=16, num_return_sequences=16)
    batch_size = 2
    return dataset_name, split, train_examples, validation_examples, search_params, batch_size


def get_xsum_beam_validation_setup():
    dataset_name = 'xsum'
    split = 'validation'
    train_examples = 1
    validation_examples = 10_000
    search_params = BeamSearchParams(num_beams=8, num_return_sequences=8)
    batch_size = 2
    return dataset_name, split, train_examples, validation_examples, search_params, batch_size


def get_xsum_psearch_train_setup():
    dataset_name = 'xsum'
    split = 'train'
    train_examples = 50_000
    validation_examples = 1
    search_params = PSearchParams(num_beams=8, num_return_sequences=8, top_p=0.9, no_repeat_ngram_size=3)
    batch_size = 2
    return dataset_name, split, train_examples, validation_examples, search_params, batch_size


def get_xsum_beam_spread_search_setup():
    dataset_name = 'xsum'
    split = 'validation'
    train_examples = 1
    validation_examples = 1200
    search_params = BeamSearchParams(num_beams=32, num_return_sequences=32)
    batch_size = 2
    return dataset_name, split, train_examples, validation_examples, search_params, batch_size
