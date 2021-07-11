from config.argument_parsing import UnsupervisedSeq2SeqTrainingArguments
from data.processing import convert_to_generation_training
from experiments.openai_dataset_reading import get_generated_gpt_dataset
from utils import compute, decorators
import datasets

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

    if training_args.use_gpt_dataset:
        old_sizes = (len(dataset['train']), len(dataset['validation']))
        gpt_dataset = get_generated_gpt_dataset()
        text_to_gpt_summary = {t: s for (t, s) in zip(gpt_dataset['article'], gpt_dataset['generated_highlights'])}
        generated_on = set(gpt_dataset['article'])
        dataset['train'] = dataset['train'].filter(lambda example: example['article'] in generated_on).map()

        dataset['validation'] = dataset['validation'].filter(lambda example: example['article'] in generated_on).map()

        print(f'old train,validation sizes:{old_sizes}, new sizes {len(dataset["train"]), len(dataset["validation"])}')
        dataset['train'] = dataset['train'].map(
            lambda example: {'generated_highlights': text_to_gpt_summary[example['article']]})
        dataset['validation'] = dataset['validation'].map(
            lambda example: {'generated_highlights': text_to_gpt_summary[example['article']]})

    # Get the column names for input/target.
    # dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)

    train_dataset, eval_dataset, predict_dataset, unsupervised_dataset = None, None, None, None
    if training_args.do_train:
        train_dataset = dataset["train"]
        if training_args.shuffle_training_set:
            train_dataset = train_dataset.shuffle(seed=training_args.shuffle_seed)
        if do_unsupervised:
            assert data_args.max_train_samples
            max_sam = data_args.max_train_samples + data_args.max_unsupervised_samples if data_args.max_unsupervised_samples else None
            train_dataset = convert_to_generation_training(train_dataset, tokenizer, data_args, max_samples=max_sam)
            splited = train_dataset.train_test_split(train_size=data_args.max_train_samples, shuffle=False)

            train_dataset, unsupervised_dataset = splited['train'], splited['test']
            if data_args.max_unsupervised_samples:
                unsupervised_dataset = unsupervised_dataset.select(range(data_args.max_unsupervised_samples))
            print(
                f'len of train dataset is {len(train_dataset)} and len of unsupervised data set {len(unsupervised_dataset)}')

        else:
            train_dataset = convert_to_generation_training(train_dataset, tokenizer, data_args,
                                                           data_args.max_train_samples)

    if training_args.do_eval:
        eval_dataset = dataset["validation"]
        eval_dataset = convert_to_generation_training(eval_dataset, tokenizer, data_args, data_args.max_eval_samples)

    if training_args.do_predict:
        predict_dataset = dataset["test"]
        predict_dataset = convert_to_generation_training(predict_dataset, tokenizer, data_args,
                                                         data_args.max_predict_samples)
    if train_dataset:
        train_dataset.name = 'train'
    if eval_dataset:
        eval_dataset.name = 'validation'
    if predict_dataset:
        predict_dataset.name = 'test'

    if do_unsupervised:
        unsupervised_dataset.name = 'unsupervised'
        return train_dataset, eval_dataset, predict_dataset, unsupervised_dataset
    return train_dataset, eval_dataset, predict_dataset


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
