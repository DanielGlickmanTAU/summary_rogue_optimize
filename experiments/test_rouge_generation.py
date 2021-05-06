from experiments import search_param_setups
from flows.flows import search_validation_loss
from flows import loading

dataset_name, split, train_examples, validation_examples, search_params, batch_size = \
    search_param_setups.get_xsum_beam_train_FULL_setup()


def _generate():
    dataset, model, tokenizer = loading.load_dataset_model_tokenizer(dataset_name, train_examples, validation_examples)
    search_validation_loss(dataset[split], model, tokenizer, search_params, batch_size)


_generate()
