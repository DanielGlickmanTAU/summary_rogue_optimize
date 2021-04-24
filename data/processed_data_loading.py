import os

from datasets import load_from_disk


def load_generated_dataset(dataset_split, model, search_params):
    mapped_search_path = get_generated_dataset_save_path(dataset_split, model, search_params)
    if os.path.isdir(mapped_search_path):
        print('loading saved dataset', mapped_search_path)
        disk = load_from_disk(mapped_search_path)


def get_generated_dataset_save_path(dataset_split, model, search_params):
    model_name = model.config.name_or_path.replace('/', '_')
    dataset_name = dataset_split.name
    ds_len = len(dataset_split)
    search_str = search_params.str_descriptor()
    mapped_search_path = '%s/processed_dataset_' % model_name + '_' + dataset_name + str(ds_len) + '_' + search_str
    return mapped_search_path
