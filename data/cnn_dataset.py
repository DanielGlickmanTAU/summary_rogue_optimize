import datasets


def get_cnn_dataset(train_subset: int = None, valid_subset: int = None, test_subset: int = None):
    dataset = datasets.load_dataset('cnn_dailymail', '3.0.0')
    if train_subset:
        dataset['train'] = dataset['train'].select(range(train_subset))
    if valid_subset:
        dataset['validation'] = dataset['validation'].select(range(valid_subset))
    if train_subset:
        dataset['test'] = dataset['test'].select(range(test_subset))
    return dataset
