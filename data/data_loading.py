from config.consts import bert_max_len
from utils import compute
import datasets

skip_constant = 6

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "newsroom": ("text", "summary"),
    "reddit_tifu": ("document", "tldr"),

}

summarization_config_mapping = {
    "cnn_dailymail": '3.0.0',
    "big_patent": 'a',
    'reddit_tifu': 'long',
    'amazon_reviews_multi': 'en'
}


def get_cnn_dataset(train_subset: int = None, valid_subset: int = None, test_subset: int = None):
    dataset = datasets.load_dataset('cnn_dailymail', '3.0.0', cache_dir=compute.get_cache_dir())
    _filter_dataset(dataset, test_subset, train_subset, valid_subset)
    set_name(dataset, 'cnn')
    return dataset


def get_xsum_dataset(train_subset: int = None, valid_subset: int = None, test_subset: int = None):
    dataset = datasets.load_dataset('xsum', cache_dir=compute.get_cache_dir())
    _filter_dataset(dataset, test_subset, train_subset, valid_subset)
    dataset.rename_column_('document', 'article')
    dataset.rename_column_('summary', 'highlights')
    dataset.remove_columns_('id')
    set_name(dataset, 'xsum')
    return dataset


def get_dataset(data_args, training_args, tokenizer):
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = datasets.load_dataset(data_args.dataset_name,
                                        summarization_config_mapping.get(data_args.dataset_name, None),
                                        cache_dir=compute.get_cache_dir())
    else:
        dataset = _load_dataset_from_file(data_args)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = dataset["test"].column_names
    else:
        raise Exception

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

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

    if training_args.do_train:
        train_dataset = dataset["train"]
        train_dataset = preprocess(column_names, data_args, preprocess_function, train_dataset,
                                   data_args.max_train_samples)

    if training_args.do_eval:
        eval_dataset = dataset["validation"]
        eval_dataset = preprocess(column_names, data_args, preprocess_function, eval_dataset,
                                  data_args.max_eval_samples)

    if training_args.do_predict:
        predict_dataset = dataset["test"]
        predict_dataset = preprocess(column_names, data_args, preprocess_function, predict_dataset,
                                     data_args.max_predict_samples)

    return train_dataset, eval_dataset, predict_dataset


def preprocess(column_names, data_args, preprocess_function, dataset_split,
               max_samples=None):
    if max_samples:
        dataset_split = dataset_split.select(range(skip_constant * max_samples))
    print(dataset_split[0])
    dataset_split = dataset_split.map(
        preprocess_function,
        batched=True,
        # batch_size=1,
        # num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    dataset_split = dataset_split.filter(
        lambda example: len(example['input_ids']) + len(example['labels']) < bert_max_len)
    if max_samples:
        # assert we have enough examples after filter.
        # this must be after select, because of .select but that does not update dataset len..
        assert len(
            dataset_split) >= max_samples, f'taking split {len(dataset_split)} for split {dataset_split.split} limited for {bert_max_len} tokens'
        dataset_split = dataset_split.select(range(max_samples))

    print(f'taking split {len(dataset_split)} for split {dataset_split.split} limited for {bert_max_len} tokens')
    return dataset_split


def _load_dataset_from_file(data_args):
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    dataset = datasets.load_dataset(extension, data_files=data_files, cache_dir=compute.get_cache_dir())
    return dataset


def set_name(dataset, name):
    dataset.name = name
    dataset['train'].name = 'train_' + name
    dataset['validation'].name = 'validation_' + name
    dataset['test'].name = 'test_' + name


def _filter_dataset(dataset, test_subset, train_subset, valid_subset):
    if train_subset:
        dataset['train'] = dataset['train'].select(range(train_subset))
    if valid_subset:
        dataset['validation'] = dataset['validation'].select(range(valid_subset))
    if test_subset:
        dataset['test'] = dataset['test'].select(range(test_subset))
