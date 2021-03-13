import datasets


def get_cnn_dataset(subset: int = None):
    dataset = datasets.load_dataset('cnn_dailymail', '3.0.0')
    if subset:
        dataset['train'] = dataset['train'].select(range(subset))
        return dataset
    return dataset
