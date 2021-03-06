from config.consts import bert_max_len
from models import tokenization_util
import nltk


def convert_to_input_ids(example, tokenizer, max_num_summaries_per_text=None, binary_classification=False,
                         include_gold=False):
    assert 'bert' in tokenizer.name_or_path, f'passed wrong tokenizer {tokenizer.name_or_path}'
    generated_highlights = example['generated_highlights'][:max_num_summaries_per_text]
    if binary_classification:
        labels = [0.] * len(generated_highlights)
    else:
        labels = example['rouge-2-all'][:max_num_summaries_per_text]

    article_list = [example['article'] for i in range(len(generated_highlights))]

    if include_gold:
        generated_highlights = [example['highlights']] + generated_highlights
        article_list = [example['article']] + article_list
        labels = [1.] + labels

    network_input = tokenization_util.tokenize(tokenizer, texts=article_list, summaries=generated_highlights)
    return {'input_ids_s': network_input['input_ids'], 'attention_mask_s': network_input['attention_mask'],
            'labels': labels}


def convert_generated_summaries_dataset_to_regression_dataset_format(dataset, tokenizer,
                                                                     max_num_summaries_per_text=None, max_seq_len=None,
                                                                     binary_classification=False,
                                                                     include_gold=False, remove_text=True):
    """

    :param dataset:
    :param tokenizer:
    :param max_num_summaries_per_text: allows limit the number of generate summaries used per text
    :param max_seq_len:
    :param binary_classification: if True, will give label 0 to generated_highlight and 1 to gold 'highights'
                                  if False, will use the rouge score for generated
    :param include_gold:  should include gold summary
    :param keep_texts: should keep the texts(article, highlights).. for training it should be false since otherwise it messed
    up the training.
    for ranking the unsupervised set, it can be true, which helps with extra tokenizations.
    :return:
    """

    dataset_map = dataset.map(
        lambda example: convert_to_input_ids(example, tokenizer, max_num_summaries_per_text, binary_classification,
                                             include_gold),
        remove_columns=list(dataset.features) if remove_text else [])
    if max_seq_len:
        len_before = len(dataset_map)
        print('WARNING! filtering sequences for only max len', max_seq_len)
        len_after = len([1 for i in range(len(dataset_map)) if len(dataset_map[i]['input_ids_s'][0]) <= 512])
        if len_after != len_before:
            dataset_map = dataset_map.filter(lambda example: len(example['input_ids_s'][0]) <= max_seq_len)
            print(f'len before filtering {len_before} , len after filtering {len(dataset_map)}')

    dataset_map.set_format(
        type="torch", columns=["input_ids_s", "attention_mask_s", "labels"], output_all_columns=not remove_text)

    return dataset_map


skip_constant = 6


def convert_to_generation_training(dataset_split, tokenizer, data_args, max_samples=None):
    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    text_column, summary_column = ("article", "highlights")

    if max_samples:
        skip_constant = 1 if not data_args.filter_examples_longer_than_max_bert_len else 8 if 'cnn' in dataset_split.builder_name else 3
        extra_to_take = skip_constant * max_samples
        # this comes to fix a bug where there is not enough examples to take. it happens in gpt-3 examples,
        # where filtering should have been done before
        # if this is not the case, it should fail in the next assert
        if len(dataset_split) > extra_to_take:
            dataset_split = dataset_split.select(range(extra_to_take))
    dataset_split = dataset_split.map(
        preprocess_function,
        batched=True,
        # batch_size=1,
        # num_proc=data_args.preprocessing_num_workers,
        # remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    if data_args.filter_examples_longer_than_max_bert_len:
        dataset_split = dataset_split.filter(
            lambda example: len(example['input_ids']) + len(example['labels']) < bert_max_len)
    if max_samples:
        # assert we have enough examples after filter.
        # this must be after select, because of .select bug that does not update dataset len..
        assert len(
            dataset_split) >= max_samples, f'taking split {len(dataset_split)} for split {dataset_split.split} limited for {bert_max_len} tokens'
        dataset_split = dataset_split.select(range(max_samples))

    print(f'taking split {len(dataset_split)} for split {dataset_split.split} limited for {bert_max_len} tokens')
    return dataset_split


def convert_dataset_with_generated_highlights_to_training_dataset(dataset, tokenizer, data_args):
    dataset = dataset.map(
        lambda example: {'highlights': example['generated_highlights'][0]},
        remove_columns=['labels']
    )
    dataset = convert_to_generation_training(dataset, tokenizer, data_args, max_samples=None)
    # huggginface magic...
    dataset.set_format(None)
    return dataset
