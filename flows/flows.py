from evaluation.evaluate import print_rouge_stuff
from experiments import experiment

from data import generated_data_loading
from models.generate import SearchParams
from models import generation


def get_generated_summaries_with_rouge(dataset_split, model, tokenizer, search_params: SearchParams, batch_size):
    mapped_search_path = generated_data_loading.get_generated_dataset_save_path(dataset_split, model, search_params)
    disk = generated_data_loading.load_generated_dataset(mapped_search_path, generation.add_rouge)
    if disk:
        return disk

    print(mapped_search_path, 'not found')
    ds = dataset_split.map(lambda x: generation.add_summary(model, tokenizer, x, search_params),
                           batched=True,
                           batch_size=batch_size)
    print('saving only summaries: saving dataset to', mapped_search_path)
    ds.save_to_disk(mapped_search_path)
    ds = ds.map(generation.add_rouge, batched=True, batch_size=batch_size)
    print('saving full: saving dataset to', mapped_search_path)
    ds.save_to_disk(mapped_search_path)
    return ds


def search_validation_loss(dataset_split, model, tokenizer, search_params: SearchParams, batch_size):
    exp = experiment.start_experiment(hyperparams={'search': search_params,
                                                   'batch_size': batch_size, 'model': model.config.name_or_path})
    ds = get_generated_summaries_with_rouge(dataset_split, model, tokenizer, search_params, batch_size)
    print_rouge_stuff(ds)
