import datasets


def get_cnn_dataset(train_subset: int = None, valid_subset: int = None, test_subset: int = None):
    dataset = datasets.load_dataset('cnn_dailymail', '3.0.0')
    _filter_dataset(dataset, test_subset, train_subset, valid_subset)
    dataset.name = 'cnn'
    return dataset


def get_xsum_dataset(train_subset: int = None, valid_subset: int = None, test_subset: int = None):
    dataset = datasets.load_dataset('xsum')
    _filter_dataset(dataset, test_subset, train_subset, valid_subset)
    dataset.rename_column_('document', 'article')
    dataset.rename_column_('summary', 'highlights')
    dataset.remove_columns_('id')
    set_name(dataset, 'xsum')
    return dataset


def set_name(dataset, name):
    dataset.name = name
    dataset['train'] = 'train_' + name
    dataset['validation'] = 'validation_' + name
    dataset['test'] = 'test_' + name


def _filter_dataset(dataset, test_subset, train_subset, valid_subset):
    if train_subset:
        dataset['train'] = dataset['train'].select(range(train_subset))
    if valid_subset:
        dataset['validation'] = dataset['validation'].select(range(valid_subset))
    if test_subset:
        dataset['test'] = dataset['test'].select(range(test_subset))
