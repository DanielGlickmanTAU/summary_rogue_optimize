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
