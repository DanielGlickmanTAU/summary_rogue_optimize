from data.generated_data_loading import load_generated_dataset
import torch

# ranker_model, tokenizer = model_loading.get_ranker_model_and_tokenizer(config)
# validation_processed_generated_xsum = generated_data_loading.load_processed_generated_dataset(
#     config.validation_mapped_saved_path, config, tokenizer)

validation_generated_xsum = load_generated_dataset(
    'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum10000_do_sampleFalse_top_pNone_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0',
    5)


def max_rouge_index(example):
    return torch.tensor(example['rouge-2-all']).argmax().item()


def diff_max_from_first(example):
    rouge_all_ = example['rouge-2-all']
    first = rouge_all_[0]
    max_ = max(rouge_all_)
    return max_ - first


def parse(example):
    max_index = max_rouge_index(example)

    best_summary = example['generated_highlights'][max_index]
    first_summary = example['generated_highlights'][0]
    best_rouge = example['rouge-2-all'][max_index]
    first_rouge = example['rouge-2-all'][0]
    return {
        'best_summary': (best_rouge, best_summary),
        'first_summary': (first_rouge, first_summary),
        'article': example['article']
    }

    # sort by diff


examples = [validation_generated_xsum[i] for i in range(len(validation_generated_xsum))]
examples = sorted(examples, key=diff_max_from_first, reverse=True)
examples = [parse(x) for x in examples]

print(validation_generated_xsum)

# 2600 / 7440 where len(best) > len(first) and best is not first
# so best is not usualy longer
