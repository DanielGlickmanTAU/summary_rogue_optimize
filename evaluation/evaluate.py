import numpy

from utils import compute

torch = compute.get_torch()


def best_at_k(labels_tensor, index_tensor, k=None):
    index_tensor = index_tensor.view(labels_tensor.shape)
    # labels_tensor = labels_tensor.float()

    if not k:
        k = labels_tensor.shape[0]
    best_indexes = index_tensor[:, 0:k].argmax(dim=1)
    labels_value_at_index = labels_tensor[torch.arange(labels_tensor.shape[0]), best_indexes]
    average_at_k = labels_tensor[torch.arange(labels_tensor.shape[0]), 0:k].mean().item()
    # print(
    #     f'results: labels_tensor: {labels_tensor} index_tensor {index_tensor} best indexes {best_indexes} label at best index {labels_value_at_index}')
    return labels_value_at_index.mean().item(), average_at_k


def print_rouge_stuff(ds):
    def avg(key): return sum(ds[key]) / len(ds[key])

    def mean_until(a, k):
        return a[:, 0:k + 1].max(axis=1).mean()

    scores = numpy.array(ds['rouge-2-all'])  # list[list[float]
    bests = [mean_until(scores, k) for k in range(len(scores[0]))]
    print('rouge-2 best at', avg('rouge-2-best'))
    print('rouge-2 avg', avg('rouge-2-avg'))
    print('rouge-2 first', avg('rouge-2-first'))
    print('rouge-2-all', bests)
