import concurrent.futures
import datasets
import numpy
import random

from utils import compute

n_threads = 20


def get_rouge(experiment_id=None):
    return datasets.load_metric('rouge',
                                experiment_id=experiment_id,
                                cache_dir=compute.get_cache_dir())


def rouge_aggregate_score_to_rouge1_mid(aggregate_score):
    return aggregate_score['rouge1'].mid.fmeasure


def rouge_aggregate_score_to_rouge2_mid(aggregate_score):
    return aggregate_score['rouge2'].mid.fmeasure


def rouge_aggregate_score_to_rougel_mid(aggregate_score):
    return aggregate_score['rougeL'].mid.fmeasure


rouge = get_rouge()


def calc_score(prediction, gold):
    if not isinstance(prediction, list) and not isinstance(gold, list):
        prediction, gold = [prediction], [gold]
    score = rouge.compute(predictions=prediction, references=gold)
    return {'rouge-1': rouge_aggregate_score_to_rouge1_mid(score),
            'rouge-2': rouge_aggregate_score_to_rouge2_mid(score)}


def calc_score_avg_and_best_and_first(predictions, gold):
    """for working with a list of predictions"""
    if not isinstance(predictions, list):
        raise Exception
    if not isinstance(gold, list):
        gold = [gold]

    # todo if this doesnt work, create a fixed list of 8 rouges, each with its own experiment_id.
    rouge = get_rouge(str(random.random()))
    scores = [rouge.compute(predictions=[pred], references=gold) for pred in predictions]
    scores = [rouge_aggregate_score_to_rouge2_mid(score) for score in scores]

    score_first = scores[0]
    score_best = max(scores)
    score_avg = sum(scores) / len(scores)
    std = numpy.std(scores)

    return {'rouge-2-best': score_best, 'rouge-2-avg': score_avg, 'rouge-2-first': score_first, 'rouge-2-all': scores,
            'rouge-2-std': std}


def calc_score_avg_best_first_for_list_of_summaries(generated_summaries, gold):
    assert len(gold) == len(generated_summaries)
    assert len(gold)
    # return [calc_score_avg_and_best_and_first(pred, ref) for pred, ref in zip(generated_summaries, gold)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        scores = executor.map(lambda x: calc_score_avg_and_best_and_first(*x), zip(generated_summaries, gold))
    return list(scores)
