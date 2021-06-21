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

import time


def get_rouge(experiment_id=None):
    if experiment_id is None:
        experiment_id = str(random.random()) + str(time.time())
    return datasets.load_metric('rouge',
                                experiment_id=experiment_id,
                                cache_dir=compute.get_cache_dir())


def rouge_aggregate_score_to_rouge1_mid(aggregate_score):
    return aggregate_score['rouge1'].mid.fmeasure


def rouge_aggregate_score_to_rouge2_mid(aggregate_score):
    return aggregate_score['rouge2'].mid.fmeasure


def rouge_aggregate_score_to_rougel_mid(aggregate_score):
    return aggregate_score['rougeL'].mid.fmeasure


rouges = ObjectPool(get_rouge, min_init=2, max_reusable=0, max_capacity=n_threads, expires=0)

total = 0.


def calc_score_avg_and_best_and_first(predictions, gold):
    """for working with a list of predictions"""
    if not isinstance(predictions, list):
        raise Exception
    if not isinstance(gold, list):
        gold = [gold]

    their_scores = [compute_rouge_from_decoder_strings(gold, [pred]) for pred in predictions]

    scores = get_by_key(their_scores, 'rouge2')
    scores1 = get_by_key(their_scores, 'rouge1')
    scoresL = get_by_key(their_scores, 'rougeL')

    score_first = scores[0]
    score_best = max(scores)
    score_avg = sum(scores) / len(scores)
    std = numpy.std(scores)

    return {'rouge-2-best': score_best, 'rouge-2-avg': score_avg, 'rouge-2-first': score_first, 'rouge-2-all': scores,
            'rouge-2-std': std,
            'rouge-1-all': scores1,
            'rouge-L-all': scoresL}


def compute_rouge_from_token_ids(preds, labels, tokenizer, ignore_pad_token_for_loss=False):
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = numpy.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    return compute_rouge_from_decoder_strings(decoded_labels, decoded_preds)


def compute_rouge_from_decoder_strings(decoded_labels, decoded_preds):
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    with rouges.get() as (rouge, _):
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return result


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def calc_score_avg_best_first_for_list_of_summaries(generated_summaries, gold):
    assert len(gold) == len(generated_summaries)
    assert len(gold)
    # return [calc_score_avg_and_best_and_first(pred, ref) for pred, ref in zip(generated_summaries, gold)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        scores = executor.map(lambda x: calc_score_avg_and_best_and_first(*x), zip(generated_summaries, gold))
    return list(scores)


def get_by_key(list_of_dicts, key):
    return [x[key] for x in list_of_dicts]
