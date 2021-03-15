import numpy
import torch
import torch.nn.functional as F


def select_best(dataset, k=0.15, scale_exponent=1., metric='rouge2', force_stop_assertions=False):
    if not force_stop_assertions:
        assert scale_exponent >= 1.  # we might want stronger ones to show more, but not the other way around
        assert k <= 0.9 or k > 1  # giving a precentile too large makes no sense
    if k < 1:
        k = int(len(dataset) * k)
    print('taking top ', k)
    weights = dataset[metric]
    index_options = list(range(0, len(dataset)))
    indexes = numpy.random.choice(population=index_options, weights=weights, k=k,
                                  replace=False)  # cant replace because using datasets.select which may use unique indexs

    return dataset.select(indexes)


def select_best(dataset, k=0.15, temp=1., metric='rouge2', ignore_assertions=False):
    if not ignore_assertions:
        # assert scale_exponent >= 1.  # we might want stronger ones to show more, but not the other way around
        assert k <= 0.9 or k >= 1  # giving a precentile too large makes no sense
    if k < 1:
        k = int(len(dataset) * k)
    k = max(k, 1)  # just for debugging on tiny datasets
    print('taking top ', k)
    # weights = [x ** scale_exponent for x in dataset[metric]]
    weights = F.softmax(torch.tensor(dataset[metric]) / temp)
    # normalizer = sum(weights)
    # weights = [x / normalizer for x in weights]
    indexes = numpy.random.choice(
        list(range(0, len(dataset))),
        k,
        p=weights.numpy(),
        replace=False  # cant replace because using datasets.select which may use unique indexs
    )
    return dataset.select(indexes)
