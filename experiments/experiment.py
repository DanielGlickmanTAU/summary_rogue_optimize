import collections.abc as collections
import dataclasses

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

    if isinstance(hyperparams, dataclasses.dataclass.__class__):
        hyperparams = dataclasses.asdict(hyperparams)
    if tags is None:
        tags = []

    experiment = Experiment(project_name='summary-sampling', workspace="danielglickmantau")
    if len(tags):
        experiment.add_tags(tags)
    if len(hyperparams):
        hyperparams = flatten(hyperparams)
        print('hyperparams:', hyperparams)
        experiment.log_parameters(hyperparams)

    return experiment
