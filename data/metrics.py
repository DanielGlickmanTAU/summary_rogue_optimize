import datasets


def get_rogue():
    return datasets.load_metric('rouge')


get_rogue()
