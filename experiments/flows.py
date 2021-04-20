import os

from datasets import load_from_disk

from data.metrics import calc_score_avg_best_first_for_list_of_summaries
from models import generate
from models.generate import SearchParams


def add_summary_and_rouge(model, tokenizer, examples, search_params: SearchParams):
    def get_by_key(list_of_dicts, key):
        return [x[key] for x in list_of_dicts]

    articles = examples['article']
    gold = examples['highlights']
    generated_summaries = generate.summarize(model, tokenizer, articles, search_params)

    num_return_sequences = search_params.num_return_sequences
    # if hack and num_return_sequences > 1:
    generated_summaries = [generated_summaries[i:i + num_return_sequences] for i in
                           range(0, len(generated_summaries), num_return_sequences)]

    scores = calc_score_avg_best_first_for_list_of_summaries(generated_summaries, gold)
    return {'rouge-2-best': get_by_key(scores, 'rouge-2-best'),
            'rouge-2-avg': get_by_key(scores, 'rouge-2-avg'),
            'rouge-2-first': get_by_key(scores, 'rouge-2-first')}


# else:
#     scores = [metrics.calc_score(pred, ref) for pred, ref in zip(generated_summaries, gold)]
#     rouge2 = [x['rouge-2'] for x in scores]
#     rouge1 = [x['rouge-1'] for x in scores]
#     return {'article': articles, 'highlights': gold, 'generated_summaries': generated_summaries,
#             'rouge2': rouge2, 'rouge1': rouge1}


def get_generated_summaries_with_rouge(dataset_split, model, tokenizer, search_params: SearchParams, batch_size):
    model_name = model.config.name_or_path.replace('/', '_')
    dataset_name = dataset_split.name
    ds_len = len(dataset_split)
    search_str = search_params.str_descriptor()

    mapped_search_path = '%s/processed_dataset_' % model_name + '_' + dataset_name + str(ds_len) + '_' + search_str
    if os.path.isdir(mapped_search_path):
        print('loading saved dataset', mapped_search_path)
        return load_from_disk(mapped_search_path)

    ds = dataset_split.map(lambda x: add_summary_and_rouge(model, tokenizer, x, search_params),
                           batched=True,
                           batch_size=batch_size)
    print('saving dataset to', mapped_search_path)
    ds.save_to_disk(mapped_search_path)
    return ds


def search_validation_loss(dataset_split, model, tokenizer, search_params: SearchParams, batch_size):
    ds = get_generated_summaries_with_rouge(dataset_split, model, tokenizer, search_params, batch_size)

    def avg(key): return sum(ds[key]) / len(ds[key])

    print('best at ', len(ds['rouge-2-best']), 'with params', search_params)
    print('rouge-2 best at', avg('rouge-2-best'))
    print('rouge-2 avg', avg('rouge-2-avg'))
    print('rouge-2 first', avg('rouge-2-first'))
