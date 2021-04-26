from models import tokenization_util


def convert_generated_summaries_dataset_to_regression_dataset_format(dataset, tokenizer, limit=None):
    def convert_to_input_ids(example):
        generated_highlights = example['generated_highlights'][:limit]
        article_list = [example['article'] for i in range(len(generated_highlights))]
        labels = example['rouge-2-all'][:limit]

        network_input = tokenization_util.tokenize(tokenizer, texts=article_list, summaries=generated_highlights)
        return {'input_ids_s': network_input['input_ids'], 'attention_mask_s': network_input['attention_mask'],
                'labels': labels}

    dataset_map = dataset.map(convert_to_input_ids,
                              remove_columns=["article", "highlights", "generated_highlights", 'rouge-2-all',
                                              'rouge-2-avg', 'rouge-2-best', 'rouge-2-first'])
    dataset_map.set_format(
        type="torch", columns=["input_ids_s", "attention_mask_s", "labels"])
    # )
    # )
    return dataset_map
