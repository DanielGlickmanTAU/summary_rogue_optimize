from data import processing
import datasets


def convert_to_regression_format(config, ranker_tokenizer, train_dataset, training_args, validation_dataset):
    validation_dataset = processing.convert_generated_summaries_dataset_to_regression_dataset_format(
        validation_dataset, ranker_tokenizer, max_num_summaries_per_text=config.num_summaries_per_text,
        max_seq_len=config.max_seq_len,
        binary_classification=config.binary_classification, include_gold=config.include_gold)
    if training_args.train_filter_on == 'train' or training_args.train_filter_on == 'both':
        train_dataset = processing.convert_generated_summaries_dataset_to_regression_dataset_format(
            train_dataset, ranker_tokenizer, max_num_summaries_per_text=config.num_summaries_per_text,
            max_seq_len=config.max_seq_len, binary_classification=config.binary_classification,
            include_gold=config.include_gold)
    if training_args.train_filter_on == 'validation' or training_args.train_filter_on == 'both':
        splited = validation_dataset.train_test_split(train_size=len(train_dataset), shuffle=False)
        train_dataset2, validation_dataset = splited['train'], splited['test']
        # assert len(validation_dataset) >= 32
        if training_args.train_filter_on == 'both':
            train_dataset = datasets.concatenate_datasets([train_dataset, train_dataset2.map()])
            train_dataset.set_format('torch')
        else:
            train_dataset = train_dataset2
    return train_dataset, validation_dataset
