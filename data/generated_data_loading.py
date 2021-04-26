import os

from datasets import load_from_disk


def load_generated_dataset(mapped_search_path, batch_size, process_function=None):
    if os.path.isdir(mapped_search_path):
        print('loading saved dataset', mapped_search_path)
        disk = load_from_disk(mapped_search_path)
        if 'rouge-2-all' not in disk.features:
            # for backwards compatibality, the 402, 32 beams on amazon
            print('adding rouge')
            disk = disk.map(process_function, batched=True, batch_size=batch_size)
            disk.save_to_disk(mapped_search_path)
        return disk
    return None


def get_generated_dataset_save_path(dataset_split, model, search_params):
    if isinstance(model, str):
        model_name = model
    else:
        model_name = model.config.name_or_path.replace('/', '_')
    dataset_name = dataset_split.name
    ds_len = len(dataset_split)
    search_str = search_params.str_descriptor()
    mapped_search_path = '%s/processed_dataset_' % model_name + '_' + dataset_name + str(ds_len) + '_' + search_str
    return mapped_search_path
