import math

from data.metrics import calc_score_avg_best_first_for_list_of_summaries
from models import generate
from models.generate import SearchParams


def add_summary_and_rouge(model, tokenizer, examples, search_params: SearchParams):
    articles = examples['article']
    gold = examples['highlights']
    if search_params.do_sample:
        # can fit like 8 beams in a time
        repeat = math.ceil(search_params.num_beams / 8)
        generated_summaries = repeat_p_search(articles, model, search_params, tokenizer, repeat=repeat)
    else:
        generated_summaries = generate.summarize(model, tokenizer, articles, search_params)

    num_return_sequences = search_params.num_return_sequences
    generated_summaries = [generated_summaries[i:i + num_return_sequences] for i in
                           range(0, len(generated_summaries), num_return_sequences)]

    # return {'generated_highlights': generated_summaries}
    scores = calc_score_avg_best_first_for_list_of_summaries(generated_summaries, gold)
    return {'rouge-2-best': get_by_key(scores, 'rouge-2-best'),
            'rouge-2-avg': get_by_key(scores, 'rouge-2-avg'),
            'rouge-2-first': get_by_key(scores, 'rouge-2-first'),
            'rouge-2-all': get_by_key(scores, 'rouge-2-all'),  # list[list[float]]
            'generated_highlights': generated_summaries}


def repeat_p_search(articles, model, search_params, tokenizer, repeat=8):
    assert search_params.num_beams % repeat == 0
    sp = search_params.clone()
    sp.num_beams = sp.num_beams // repeat
    sp.num_return_sequences = sp.num_return_sequences // repeat
    generated_summaries_lists = [generate.summarize(model, tokenizer, articles, sp) for i in range(repeat)]
    generated_summaries = []
    num_generated_summarizes = len(generated_summaries_lists[0])
    for summary_index in range(num_generated_summarizes):
        for summaries_list in generated_summaries_lists:
            generated_summaries.append(summaries_list[summary_index])
    return generated_summaries


def get_by_key(list_of_dicts, key):
    return [x[key] for x in list_of_dicts]