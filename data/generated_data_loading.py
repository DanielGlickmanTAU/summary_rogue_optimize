import os

from datasets import load_from_disk

from config.config import RankingDatasetConfig
from data import processing
from models import generation
from models.generate import SearchParams


def load_generated_dataset(mapped_search_path, process_function=None):
    if os.path.isdir(mapped_search_path):
        print('loading saved dataset', mapped_search_path)
        disk = load_from_disk(mapped_search_path)
        if process_function and 'rouge-2-all' not in disk.features:
            # for backwards compatibality
            print('adding rouge')
            disk = disk.map(process_function, batched=True, batch_size=20)
            disk.save_to_disk(mapped_search_path)
        return disk
    return None


def load_processed_generated_dataset(validation_mapped_saved_path, config: RankingDatasetConfig, tokenizer,
                                     max_examples=None, binary_classification=False, include_gold=False):
    validation_generated_xsum = load_generated_dataset(validation_mapped_saved_path)
    # validation_generated_xsum = _limit_before_processing(config, validation_generated_xsum)

    validation_processed_generated_xsum = processing.convert_generated_summaries_dataset_to_regression_dataset_format(
        validation_generated_xsum, tokenizer, max_num_summaries_per_text=config.num_summaries_per_text,
        max_seq_len=config.max_seq_len, binary_classification=binary_classification, include_gold=include_gold)

    validation_processed_generated_xsum = _limit_after_processing(validation_processed_generated_xsum,
                                                                  max_examples=max_examples)

    return validation_processed_generated_xsum


def _limit_after_processing(validation_processed_generated_xsum, max_examples):
    if max_examples:
        if len(validation_processed_generated_xsum) < max_examples:
            print(f'WARNING not enough examples, only {len(validation_processed_generated_xsum)}')
            return validation_processed_generated_xsum
        validation_processed_generated_xsum = validation_processed_generated_xsum.select(range(max_examples))
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


@DeprecationWarning
def get_generated_summaries_with_rouge(dataset_split, model, tokenizer, search_params: SearchParams, batch_size):
    mapped_search_path = get_generated_dataset_save_path(dataset_split, model, search_params)
    disk = load_generated_dataset(mapped_search_path, generation.add_rouge)
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


def get_generated_summaries(dataset_split, model, tokenizer, search_params: SearchParams, batch_size,
                            load_generated):
    """
    :param load_generated: if True, will try to load dataset of generated that is already saved if exists.
    if does not exist, will generate one and save it.

    Todo: need to take into account the random seed of the model, because models trained with different splits will
    have different results.
    However it already takes into account the models training size, learning rate etc, as it is part of the model_path_or_name.
    Can probably be fixed by adding the random split to model_path_of_name
    :return:
    """
    if load_generated:
        mapped_search_path = get_generated_dataset_save_path(dataset_split, model, search_params)
        disk = load_generated_dataset(mapped_search_path)
        if disk:
            return disk
        print(mapped_search_path, 'not found')
    print('generating summaries')
    ds = dataset_split.map(lambda x: generation.add_summary(model, tokenizer, x, search_params),
                           batched=True,
                           batch_size=batch_size)
    if load_generated:
        print('saving only summaries: saving dataset to', mapped_search_path)
        ds.save_to_disk(mapped_search_path)

    return ds
