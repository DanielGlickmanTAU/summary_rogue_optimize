from utils import compute, decorators
import datasets
import json
import collections

data = json.load(open('results_open_json.json'))

d = collections.defaultdict(list)
for x in data:
    d['article'].append(x['text'])
    d['generated_highlights'].append([x['summary']])

data = datasets.Dataset.from_dict(d)
