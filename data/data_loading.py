from config.argument_parsing import UnsupervisedSeq2SeqTrainingArguments
from config.consts import bert_max_len
from utils import compute, decorators
import datasets

skip_constant = 6

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "xsum": ("document", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
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


@decorators.measure_time
def get_dataset(data_args, training_args: UnsupervisedSeq2SeqTrainingArguments, tokenizer, do_unsupervised=False):
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    dataset_name = data_args.dataset_name
    if dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = datasets.load_dataset(dataset_name,
                                        summarization_config_mapping.get(dataset_name, None),
                                        cache_dir=compute.get_cache_dir())
    else:
        dataset = _load_dataset_from_file(data_args)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    for t in ["description", "text", "document", "extract_text"]:
        if t in dataset['train'].column_names:
            dataset.rename_column_(t, "article")
    for s in ["abstract", "summary", "summary_text"]:
        if s in dataset['train'].column_names:
            dataset.rename_column_(s, "highlights")

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
    # dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)

    train_dataset, eval_dataset, predict_dataset, unsupervised_dataset = None, None, None, None
    if training_args.do_train:
        train_dataset = dataset["train"]
        if training_args.shuffle_training_set:
            train_dataset = train_dataset.shuffle(seed=42)
        if do_unsupervised:
            max_sam = None
            train_dataset = convert_to_generation_training(data_args, train_dataset, tokenizer, max_samples=max_sam)
            splited = train_dataset.train_test_split(train_size=data_args.max_train_samples, shuffle=False)

            train_dataset, unsupervised_dataset = splited['train'], splited['test']
            if data_args.max_unsupervised_samples:
                unsupervised_dataset = unsupervised_dataset.select(range(data_args.max_unsupervised_samples))
            print(
                f'len of train dataset is {len(train_dataset)} and len of unsupervised data set {len(unsupervised_dataset)}')

        else:
            train_dataset = convert_to_generation_training(data_args, train_dataset, tokenizer,
                                                           data_args.max_train_samples)

    if training_args.do_eval:
        eval_dataset = dataset["validation"]
        eval_dataset = convert_to_generation_training(data_args, eval_dataset, tokenizer,
                                                      data_args.max_eval_samples)

    if training_args.do_predict:
        predict_dataset = dataset["test"]
        predict_dataset = convert_to_generation_training(data_args, predict_dataset, tokenizer,
                                                         data_args.max_predict_samples)

    if do_unsupervised:
        unsupervised_dataset.name = 'unsupervised'
        return train_dataset, eval_dataset, predict_dataset, unsupervised_dataset
    return train_dataset, eval_dataset, predict_dataset


def convert_to_generation_training(data_args, dataset_split, tokenizer, max_samples=None):
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
        dataset_split = dataset_split.select(range(skip_constant * max_samples))
    dataset_split = dataset_split.map(
        preprocess_function,
        batched=True,
        # batch_size=1,
        # num_proc=data_args.preprocessing_num_workers,
        # remove_columns=column_names,
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
