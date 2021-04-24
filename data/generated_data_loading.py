import os

from datasets import load_from_disk

from data.metrics import calc_score_avg_best_first_for_list_of_summaries


def load_generated_dataset(mapped_search_path, batch_size):
    if os.path.isdir(mapped_search_path):
        print('loading saved dataset', mapped_search_path)
        disk = load_from_disk(mapped_search_path)
        if 'rouge-2-all' not in disk.features:
            # for backwards compatibality, the 402, 32 beams on amazon
            disk = disk.map(_add_scores, batched=True, batch_size=batch_size)
            disk.save_to_disk(mapped_search_path)
        return disk
    return None


def get_generated_dataset_save_path(dataset_split, model, search_params):
    model_name = model.config.name_or_path.replace('/', '_')
    dataset_name = dataset_split.name
    ds_len = len(dataset_split)
    search_str = search_params.str_descriptor()
    mapped_search_path = '%s/processed_dataset_' % model_name + '_' + dataset_name + str(ds_len) + '_' + search_str
    return mapped_search_path


def _add_scores(examples):
    gold = examples['highlights']
    generated_summaries = examples['generated_highlights']
    scores = calc_score_avg_best_first_for_list_of_summaries(generated_summaries, gold)
    return {'rouge-2-all': [x['rouge-2-all'] for x in scores]}
