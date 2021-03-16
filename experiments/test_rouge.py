import experiment
from data import data_loading, metrics
from models import model_loading, generate
from models.candidate_selection import select_best
from train import training
import random


def get_random_examples(ds, k):
    indexes = random.sample(range(len(ds)), k)
    return ds.select(indexes)


def add_summary_and_rouge(examples):
    articles = examples['article']
    gold = examples['highlights']
    generated_summaries = generate.summarize(model, tokenizer, articles)

    assert len(gold) == len(generated_summaries)
    scores = [metrics.calc_score(pred, ref) for pred, ref in zip(generated_summaries, gold)]
    rouge2 = [x['rouge-2'] for x in scores]
    rouge1 = [x['rouge-1'] for x in scores]

    return {'generated_summaries': generated_summaries, 'rouge2': rouge2, 'rouge1': rouge1}


def eval_metric(dataset_split, exp):
    ds = dataset_split.map(add_summary_and_rouge, batched=True, batch_size=batch_size)
    ds_rouge_2 = sum(ds['rouge2']) / len(ds['rouge2'])
    ds_rouge_1 = sum(ds['rouge1']) / len(ds['rouge1'])
    print('rouge2 is ', ds_rouge_2, ' evaluate on', len(ds['rouge2']))
    try:
        exp.log_metrics({'rouge1': ds_rouge_1, 'rouge2': ds_rouge_2})
    except Exception:
        pass
    return ds_rouge_2


rouge = metrics.get_rouge()


def do_experiment(model, tokenizer, cnn, train_examples, examples_for_training_epoch, learning_rate, temperature,
                  precentile,
                  do_sample, top_p, num_beams, strikes=3):
    exp = experiment.start_experiment(hyperparams={
        'batch_size': batch_size,
        'train_examples': train_examples,
        'validation_examples': validation_examples,
        'temperature': temperature,
        'precentile': precentile,
        'strikes': strikes,
        'top_p': top_p,
        'top_k': generate.top_k,
        'do_sample': do_sample,
        'num_beams': num_beams,
        'num_return_sequences': generate.num_return_sequences,
        'model_name': model_loading.model_name,
        'examples_for_training_batch': examples_for_training_epoch,
        'learning_rate': training.learning_rate,
        'validation_split': validation_split
    })

    cnn_train = cnn['train']
    test_summaries = get_random_examples(cnn_train, examples_for_training_epoch).map(add_summary_and_rouge,
                                                                                     batched=True,
                                                                                     batch_size=batch_size)
    current_valid_score = eval_metric(cnn[validation_split], exp)
    while True:
        print('selecting top')
        top = select_best(test_summaries, temp=temperature, k=precentile)
        print('top 3: ', top[:3])
        # replace gold tags with generated
        # comment this out when I want to compare to normal train.. and also set select scale_exp=0
        top = top.map(lambda examples: {'highlights': examples['generated_summaries']})
        print('train')
        training.train(model, tokenizer, top, int(batch_size / 2), learning_rate=learning_rate)

        new_valid_score = eval_metric(cnn[validation_split], exp)
        if new_valid_score <= current_valid_score:
            strikes = strikes - 1
            if strikes <= 0:
                break

        current_valid_score = new_valid_score
        test_summaries = get_random_examples(cnn_train, examples_for_training_epoch).map(add_summary_and_rouge,
                                                                                         batched=True,
                                                                                         batch_size=batch_size)
    print('done single expirment')
    exp.end()


validation_split = 'validation'

batch_size = 16
train_examples = 50_000
examples_for_training_epoch = 3_200
examples_for_training_epoch = train_examples
# train_examples = batch_size * 1
validation_examples = 250
# validation_examples = batch_size * 1

# validation_split = 'train'
# train_examples = 100
# examples_for_training_batch = 100

strikes = 3
temperature = 2.5
precentile = 0.06

# examples_for_training_batch = 320

model, tokenizer = model_loading.get_bart_model_and_tokenizer()
cnn = data_loading.get_xsum_dataset(train_subset=train_examples, valid_subset=validation_examples)

do_experiment(model, tokenizer, cnn,
              train_examples=1_000,
              examples_for_training_epoch=640,
              learning_rate=1e-07,
              temperature=2.5,
              precentile=0.05,
              do_sample=False, top_p=None, num_beams=4)

do_experiment(model, tokenizer, cnn,
              train_examples=1_000,
              examples_for_training_epoch=640,
              learning_rate=1e-07,
              temperature=0.7,
              precentile=0.15,
              do_sample=False, top_p=None, num_beams=4)

do_experiment(model, tokenizer, cnn,
              train_examples=10_000,
              examples_for_training_epoch=3200,
              learning_rate=1e-07,
              temperature=2.5,
              precentile=0.06,
              do_sample=False, top_p=None, num_beams=4)

do_experiment(model, tokenizer, cnn,
              train_examples=10_000,
              examples_for_training_epoch=3200,
              learning_rate=1e-07,
              temperature=3.2,
              precentile=0.04,
              do_sample=False, top_p=None, num_beams=4,
              strikes=2
              )

do_experiment(model, tokenizer, cnn,
              train_examples=1_000,
              examples_for_training_epoch=640,
              learning_rate=1e-07,
              temperature=2.5,
              precentile=0.05,
              do_sample=True, top_p=0.9, num_beams=4)

do_experiment(model, tokenizer, cnn,
              train_examples=1_000,
              examples_for_training_epoch=640,
              learning_rate=1e-07,
              temperature=0.7,
              precentile=0.15,
              do_sample=True, top_p=0.9, num_beams=4)
