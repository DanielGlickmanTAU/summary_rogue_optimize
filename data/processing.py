from data.SampleInput import SampleInput
from models import tokenize


def convert_generated_summaries_dataset_to_regression_dataset_format(dataset, tokenizer):
    def convert_to_input_ids(example):
        generated_highlights = example['generated_highlights']
        article_list = [example['article'] for i in range(len(generated_highlights))]

        network_input = tokenize.tokenize(tokenizer, texts=article_list, summaries=generated_highlights)
        custom_input = SampleInput(network_input['input_ids'], network_input['attention_mask'])
        return {'custom_input': network_input['input_ids'], 'labels': example['rouge-2-all']}

    dataset_map = dataset.map(convert_to_input_ids)
    # possibly remove redundant columns here
    return dataset_map
