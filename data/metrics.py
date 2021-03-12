import datasets


def get_rouge():
    return datasets.load_metric('rouge')


def rouge_aggregate_score_to_rouge1_mid(aggregate_score):
    return aggregate_score['rouge1'].mid.fmeasure


def rouge_aggregate_score_to_rouge2_mid(aggregate_score):
    return aggregate_score['rouge2'].mid.fmeasure
