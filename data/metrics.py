import concurrent.futures
import datasets
import numpy
import random
from object_pool import ObjectPool

from filelock import FileLock

from utils import compute

import nltk  # Here to have a nice missing dependency error message early on

try:
    nltk.data.find("tokenizers/punkt",
                   # paths=[compute.get_cache_dir()]
                   )
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True,
                      download_dir='/home/yandex/AMNLP2021/glickman1/anaconda3/envs/comet/nltk_data')

n_threads = 20


def get_rouge(experiment_id=None):
    if experiment_id is None:
        experiment_id = str(random.random())
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

rouges = ObjectPool(get_rouge, min_init=2, max_reusable=0, max_capacity=n_threads)


def calc_score(prediction, gold):
    if not isinstance(prediction, list) and not isinstance(gold, list):
        prediction, gold = [prediction], [gold]
    score = rouge.compute(predictions=prediction, references=gold)
    return {'rouge-1': rouge_aggregate_score_to_rouge1_mid(score),
            'rouge-2': rouge_aggregate_score_to_rouge2_mid(score)}


import time
import threading

total = 0.


def calc_score_avg_and_best_and_first(predictions, gold):
    """for working with a list of predictions"""
    if not isinstance(predictions, list):
        raise Exception
    if not isinstance(gold, list):
        gold = [gold]

    # todo if this doesnt work, create a fixed list of 8 rouges, each with its own experiment_id.
    global total
    t = time.time()
    with rouges.get() as (rouge, _):
        # rouge = get_rouge(str(threading.get_ident()))
        total += time.time() - t
        if total > 10:
            total = 0
            for i in range(1000):
                print('creating rouge taking lots of time')

        scores = [rouge.compute(predictions=[pred], references=gold, use_stemmer=True) for pred in predictions]
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
