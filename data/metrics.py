import datasets


def get_rouge():
    return datasets.load_metric('rouge')


def rouge_aggregate_score_to_rouge1_mid(aggregate_score):
    return aggregate_score['rouge1'].mid.fmeasure


def rouge_aggregate_score_to_rouge2_mid(aggregate_score):
    return aggregate_score['rouge2'].mid.fmeasure


def rouge_aggregate_score_to_rougel_mid(aggregate_score):
    return aggregate_score['rougeL'].mid.fmeasure


rouge = get_rouge()


def calc_score(prediction, gold):
    score = rouge.compute(predictions=[prediction], references=[gold])
    return {'rouge-1': rouge_aggregate_score_to_rouge1_mid(score),
            'rouge-2': rouge_aggregate_score_to_rouge2_mid(score)}
