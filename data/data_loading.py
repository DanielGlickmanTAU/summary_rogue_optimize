from utils import compute
import datasets


def get_cnn_dataset(train_subset: int = None, valid_subset: int = None, test_subset: int = None):
    dataset = datasets.load_dataset('cnn_dailymail', '3.0.0', cache_dir=compute.get_cache_dir())
    _filter_dataset(dataset, test_subset, train_subset, valid_subset)
    set_name(dataset, 'cnn')
    return dataset


def get_xsum_dataset(train_subset: int = None, valid_subset: int = None, test_subset: int = None):
    dataset = datasets.load_dataset('xsum', cache_dir=compute.get_cache_dir())
    _filter_dataset(dataset, test_subset, train_subset, valid_subset)
    dataset.rename_column_('document', 'article')
    dataset.rename_column_('summary', 'highlights')
    dataset.remove_columns_('id')
    set_name(dataset, 'xsum')
    return dataset


def set_name(dataset, name):
    dataset.name = name
    dataset['train'].name = 'train_' + name
    dataset['validation'].name = 'validation_' + name
    dataset['test'].name = 'test_' + name


def _filter_dataset(dataset, test_subset, train_subset, valid_subset):
    if train_subset:
        dataset['train'] = dataset['train'].select(range(train_subset))
    if valid_subset:
        dataset['validation'] = dataset['validation'].select(range(valid_subset))
    if test_subset:
        dataset['test'] = dataset['test'].select(range(test_subset))
