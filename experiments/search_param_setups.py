from models.generate import BeamSearchParams


def get_xsum_beam_train_config():
    dataset_name = 'xsum'
    split = 'train'
    train_examples = 50_000
    validation_examples = 1
    search_params = BeamSearchParams(num_beams=32, num_return_sequences=32)
    batch_size = 2
