import os

from datasets import load_from_disk

from config.config import RankingDatasetConfig
from data import processing


def load_generated_dataset(mapped_search_path, batch_size, process_function=None):
    if os.path.isdir(mapped_search_path):
        print('loading saved dataset', mapped_search_path)
        disk = load_from_disk(mapped_search_path)
        if 'rouge-2-all' not in disk.features:
            # for backwards compatibality
            print('adding rouge')
            disk = disk.map(process_function, batched=True, batch_size=20)
            disk.save_to_disk(mapped_search_path)
        return disk
    return None


def load_processed_generated_dataset(validation_mapped_saved_path, config: RankingDatasetConfig, tokenizer):
    validation_generated_xsum = load_generated_dataset(validation_mapped_saved_path, batch_size=5)
    # validation_generated_xsum = _limit_before_processing(config, validation_generated_xsum)

    validation_processed_generated_xsum = processing.convert_generated_summaries_dataset_to_regression_dataset_format(
        validation_generated_xsum, tokenizer, max_num_summaries_per_text=config.num_summaries_per_text,
        max_seq_len=config.max_seq_len)

    validation_processed_generated_xsum = _limit_after_processing(config, validation_processed_generated_xsum)

    return validation_processed_generated_xsum


def _limit_after_processing(config, validation_processed_generated_xsum):
    if config.num_examples:
        if len(validation_processed_generated_xsum) < config.num_examples:
            print(f'WARNING not enough examples, only {len(validation_processed_generated_xsum)}')
            return validation_processed_generated_xsum
        validation_processed_generated_xsum = validation_processed_generated_xsum.select(range(config.num_examples))
    return validation_processed_generated_xsum


# unused
def _limit_before_processing(config, validation_generated_xsum):
    if config.num_examples:
        num_examples_to_process = int(2.5 * (config.num_skip + config.num_examples))
        validation_generated_xsum = validation_generated_xsum.select(
            range(config.num_skip, max(num_examples_to_process, len(validation_generated_xsum))))
    return validation_generated_xsum


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
