from data.generated_data_loading import get_generated_summaries_with_rouge
from evaluation.evaluate import print_rouge_stuff
from experiments import experiment

from models.generate import SearchParams


def search_validation_loss(dataset_split, model, tokenizer, search_params: SearchParams, batch_size):
    exp = experiment.start_experiment(hyperparams={'search': search_params,
                                                   'batch_size': batch_size, 'model': model.config.name_or_path})
    ds = get_generated_summaries_with_rouge(dataset_split, model, tokenizer, search_params, batch_size)
    print_rouge_stuff(ds)
