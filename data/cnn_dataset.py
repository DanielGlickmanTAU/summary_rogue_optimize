import datasets


def get_cnn_dataset():
    return datasets.load_dataset('cnn_dailymail', '3.0.0')