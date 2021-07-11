import collections.abc as collections
import dataclasses
import os
import comet_ml
from comet_ml import Experiment


def start_experiment(tags=None, hyperparams=None):
    def flatten(d):
        items = []
        for k, v in d.items():
            if isinstance(v, collections.MutableMapping):
                if hasattr(v, '__dict__'):
                    items.extend(flatten(v.__dict__).items())
                else:
                    items.extend(flatten(v).items())
            else:
                items.append((k, v))
        return dict(items)

    if hyperparams is None:
        hyperparams = {}
    if isinstance(hyperparams, list):
        h = {}
        for h2 in hyperparams:
            h.update(dataclasses.asdict(h2))
        hyperparams = h
    if isinstance(hyperparams, dataclasses.dataclass.__class__):
        hyperparams = dataclasses.asdict(hyperparams)
    if tags is None:
        tags = []

    experiment = Experiment(project_name='feedback-filter', workspace="danielglickmantau")
    if len(tags):
        experiment.add_tags(tags)
    if len(hyperparams):
        hyperparams = flatten(hyperparams)
        experiment.log_parameters(hyperparams)

    print(os.popen(f'squeue  | grep glickman').read())
    if len(hyperparams):
        print('hyperparams:', flatten(hyperparams))
    return experiment


def log_metrics(metrics):
    try:
        experiment = comet_ml.config.get_global_experiment()
        experiment.log_metrics(metrics)
    except:
        print('WARNING FAILED REPORTING METRICS', metrics)
