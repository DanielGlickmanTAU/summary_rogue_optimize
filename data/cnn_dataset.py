import datasets


def get_cnn_dataset(subset: int = None):
    if subset:
        return datasets.load_dataset('cnn_dailymail', '3.0.0').select(range(subset))
    return datasets.load_dataset('cnn_dailymail', '3.0.0')
