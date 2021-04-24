import numpy

from data import processed_data_loading
from experiments import experiment
from models.generate import SearchParams
from models import generation


def get_generated_summaries_with_rouge(dataset_split, model, tokenizer, search_params: SearchParams, batch_size):
    mapped_search_path = processed_data_loading.get_generated_dataset_save_path(dataset_split, model, search_params)
    disk = processed_data_loading.load_generated_dataset(mapped_search_path, batch_size)
    if disk:
        return disk

    print(mapped_search_path, 'not found')
    ds = dataset_split.map(lambda x: generation.add_summary_and_rouge(model, tokenizer, x, search_params),
                           batched=True,
                           batch_size=batch_size)
    print('saving dataset to', mapped_search_path)
    ds.save_to_disk(mapped_search_path)
    return ds


def search_validation_loss(dataset_split, model, tokenizer, search_params: SearchParams, batch_size):
    exp = experiment.start_experiment(hyperparams={'search': search_params,
                                                   'batch_size': batch_size, 'model': model.config.name_or_path})
    ds = get_generated_summaries_with_rouge(dataset_split, model, tokenizer, search_params, batch_size)

    def avg(key): return sum(ds[key]) / len(ds[key])

    def mean_until(a, k):
        return a[:, 0:k + 1].max(axis=1).mean()

    scores = numpy.array(ds['rouge-2-all'])  # list[list[float]
    bests = [mean_until(scores, k) for k in range(len(scores[0]))]
    print('best at ', len(ds['rouge-2-best']), 'with params', search_params)
    print('rouge-2 best at', avg('rouge-2-best'))
    print('rouge-2 avg', avg('rouge-2-avg'))
    print('rouge-2 first', avg('rouge-2-first'))
    print('rouge-2-all', bests)
    # print('rouge-2-std average', avg('rouge-2-std'))
