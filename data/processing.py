from models import tokenization_util
import nltk


def convert_generated_summaries_dataset_to_regression_dataset_format(dataset, tokenizer,
                                                                     max_num_summaries_per_text=None, max_seq_len=None):
    def convert_to_input_ids(example):
        generated_highlights = example['generated_highlights'][:max_num_summaries_per_text]
        article_list = [example['article'] for i in range(len(generated_highlights))]
        labels = example['rouge-2-all'][:max_num_summaries_per_text]

        network_input = tokenization_util.tokenize(tokenizer, texts=article_list, summaries=generated_highlights)
        return {'input_ids_s': network_input['input_ids'], 'attention_mask_s': network_input['attention_mask'],
                'labels': labels}

    dataset_map = dataset.map(convert_to_input_ids,
                              remove_columns=list(dataset.features))
    if max_seq_len:
        len_before = len(dataset_map)
        print('WARNING! filtering sequences for only max len', max_seq_len)
        dataset_map = dataset_map.filter(lambda example: len(example['input_ids_s'][0]) < max_seq_len)
        print(f'len before filtering {len_before} , len after filtering {len(dataset_map)}')

    dataset_map.set_format(
        type="torch", columns=["input_ids_s", "attention_mask_s", "labels"])

    return dataset_map
