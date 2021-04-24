from data import processed_data_loading
from experiments import experiment
import os

import numpy
from datasets import load_from_disk

from data.metrics import calc_score_avg_best_first_for_list_of_summaries
from models import generate
from models.generate import SearchParams
import math


def get_by_key(list_of_dicts, key):
    return [x[key] for x in list_of_dicts]


def add_summary_and_rouge(model, tokenizer, examples, search_params: SearchParams):
    articles = examples['article']
    gold = examples['highlights']
    if search_params.do_sample:
        # can fit like 8 beams in a time
        repeat = math.ceil(search_params.num_beams / 8)
        generated_summaries = repeat_p_search(articles, model, search_params, tokenizer, repeat=repeat)
    else:
        generated_summaries = generate.summarize(model, tokenizer, articles, search_params)

    num_return_sequences = search_params.num_return_sequences
    generated_summaries = [generated_summaries[i:i + num_return_sequences] for i in
                           range(0, len(generated_summaries), num_return_sequences)]

    # return {'generated_highlights': generated_summaries}
    scores = calc_score_avg_best_first_for_list_of_summaries(generated_summaries, gold)
    return {'rouge-2-best': get_by_key(scores, 'rouge-2-best'),
            'rouge-2-avg': get_by_key(scores, 'rouge-2-avg'),
            'rouge-2-first': get_by_key(scores, 'rouge-2-first'),
            'rouge-2-all': get_by_key(scores, 'rouge-2-all'),  # list[list[float]]
            'generated_highlights': generated_summaries}


def repeat_p_search(articles, model, search_params, tokenizer, repeat=8):
    assert search_params.num_beams % repeat == 0
    sp = search_params.clone()
    sp.num_beams = sp.num_beams // repeat
    sp.num_return_sequences = sp.num_return_sequences // repeat
    generated_summaries_lists = [generate.summarize(model, tokenizer, articles, sp) for i in range(repeat)]
    generated_summaries = []
    num_generated_summarizes = len(generated_summaries_lists[0])
    for summary_index in range(num_generated_summarizes):
        for summaries_list in generated_summaries_lists:
            generated_summaries.append(summaries_list[summary_index])
    return generated_summaries


def get_generated_summaries_with_rouge(dataset_split, model, tokenizer, search_params: SearchParams, batch_size):
    mapped_search_path = processed_data_loading.get_generated_dataset_save_path(dataset_split, model, search_params)
    if os.path.isdir(mapped_search_path):
        print('loading saved dataset', mapped_search_path)
        disk = load_from_disk(mapped_search_path)
        if 'rouge-2-all' not in disk.features:
            # for backwards compatibality, the 402, 32 beams on amazon
            disk = disk.map(add_scores, batched=True, batch_size=batch_size)
            disk.save_to_disk(mapped_search_path)
        return disk
    print(mapped_search_path, 'not found')
    ds = dataset_split.map(lambda x: add_summary_and_rouge(model, tokenizer, x, search_params),
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


def add_scores(examples):
    gold = examples['highlights']
    generated_summaries = examples['generated_highlights']
    scores = calc_score_avg_best_first_for_list_of_summaries(generated_summaries, gold)
    return {'rouge-2-all': get_by_key(scores, 'rouge-2-all')}
