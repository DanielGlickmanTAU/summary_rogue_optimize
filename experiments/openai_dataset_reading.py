from utils import compute, decorators
import datasets
import json
import collections


def clean_summary(param):
    return param.replace('\n', '')


def get_generated_gpt_dataset():
    data = json.load(open('results_open_json.json'))
    d = collections.defaultdict(list)
    for x in data:
        d['article'].append(x['text'])
        d['generated_highlights'].append([clean_summary(x['summary'])])
    return datasets.Dataset.from_dict(d)
