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

    def print_for_rouge_type(ds, rouge_type, all=True):
        scores = numpy.array(ds[('rouge-%s-all' % rouge_type)])  # list[list[float]
        bests = [mean_until(scores, k) for k in range(len(scores[0]))]
        if all:
            print('rouge-%s best at' % rouge_type, avg('rouge-%s-best' % rouge_type))
            print('rouge-%s avg' % rouge_type, avg('rouge-%s-avg' % rouge_type))
            print('rouge-%s first' % rouge_type, avg('rouge-%s-first' % rouge_type))
        print('rouge-%s-all' % rouge_type, bests)

    rouge_type = '2'
    print_for_rouge_type(ds, rouge_type)
    print_for_rouge_type(ds, '1', all=False)
    print_for_rouge_type(ds, 'L', all=False)
    return avg('rouge-%s-first' % rouge_type)
