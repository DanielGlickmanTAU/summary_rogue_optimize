from utils import decorators, compute
from transformers import TrainingArguments
import datasets

from config.argument_parsing import parse_generation_args
from config.config import RankerConfig
from data import data_loading, generated_data_loading, processing
from data.processing import convert_dataset_with_generated_highlights_to_training_dataset
from evaluation import evaluate
from experiments import experiment
from experiments.experiment import log_metrics
from models import model_loading, generation, checkpoints
from time import time
import random

# TODO this also exists in run_summarization, origanize and move this to one place
from models.generate import BeamSearchParams
from train import generation_training, training


@decorators.measure_time
def my_eval(dataset, model, tokenizer, search_params, description=''):
    ds = generation.add_summary_and_rouge(model, tokenizer, dataset,
                                          search_params)
    print(f'evaluate {description}')
    return evaluate.print_rouge_stuff(ds)


@decorators.measure_time
def do_train(model, tokenizer, train_dataset, eval_dataset, training_args, data_args, last_checkpoint):
    # need this because the trainer remove features that are not neccessery for the model(like article and highlights), which messes things up later.
    train_dataset = train_dataset.map()
    eval_dataset = eval_dataset.map()
    trainer = generation_training.create_trainer(train_dataset, eval_dataset, training_args, data_args, model,
                                                 tokenizer)

    checkpoint = last_checkpoint
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    # elif last_checkpoint is not None:
    #     checkpoint = last_checkpoint
    if checkpoint:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
    else:
        train_result = trainer.train()

    if training_args.save_model_after_train:
        t = time()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()
        print(f'saving took {time() - t} seconds')
    else:
        print('skiping saving generation model')
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    return metrics


data_args, model_args, training_args, last_checkpoint = parse_generation_args()
search_params = BeamSearchParams(num_return_sequences=1, num_beams=data_args.num_beams)

model_checkpoint = \
    checkpoints.get_checkpoint_output_dir(data_args.dataset_name, model_args.model_name_or_path,
                                          data_args.max_train_samples, training_args.learning_rate,
                                          extra=training_args.shuffle_seed if training_args.shuffle_training_set else None)
training_args.output_dir = model_checkpoint + str(random.random())

experiment.start_experiment(hyperparams=[data_args, training_args, model_args],
                            tags=[] if training_args.track_experiment else ['debug'])

if training_args.load_generated_model:
    if training_args.shuffle_training_set:
        raise NotImplementedError('it is not supported right now with get_generated_summaries')
    model_args.model_name_or_path = model_checkpoint
    model, tokenizer = model_loading.get_model_and_tokenizer(model_args)
else:
    model, tokenizer = model_loading.get_model_and_tokenizer(model_args)

train_dataset, eval_dataset, predict_dataset, unsupervised_data = data_loading.get_dataset(data_args, training_args,
                                                                                           tokenizer,
                                                                                           do_unsupervised=True)

if not training_args.load_generated_model:
    do_train(model, tokenizer, train_dataset, eval_dataset, training_args, data_args, last_checkpoint)

# eval on train set(to see overfit)
if training_args.eval_also_on_train_first_time:
    my_eval(train_dataset, model, tokenizer, search_params,
            f'on TRAIN set after training on {len(train_dataset)} samples')

# eval on test test
rouge_on_test = None
if not training_args.skip_first_test_eval:
    rouge_on_test = my_eval(predict_dataset, model, tokenizer, search_params,
                            f'on TEST set after training on {len(train_dataset)} samples')
    log_metrics({'rouge2_on_test': rouge_on_test})

# here do if training_args.use_gpt_dataset load from my new python file... else below
if training_args.use_gpt_dataset:
    unsupervised_data = generated_data_loading.get_generated_summaries(unsupervised_data, model, tokenizer,
                                                                       search_params,
                                                                       batch_size=training_args.per_device_eval_batch_size,
                                                                       load_generated=training_args.load_generated_model)

    train_dataset = generated_data_loading.get_generated_summaries(train_dataset, model, tokenizer,
                                                                   search_params,
                                                                   batch_size=training_args.per_device_eval_batch_size,
                                                                   load_generated=training_args.load_generated_model)

    eval_dataset = generated_data_loading.get_generated_summaries(eval_dataset, model, tokenizer,
                                                                  search_params,
                                                                  batch_size=training_args.per_device_eval_batch_size,
                                                                  load_generated=training_args.load_generated_model)


def rank(unsupervised_data, train_dataset, validation_dataset, training_args, prediction_dataset=None):
    ranking = training_args.ranking
    if ranking == 'oracle':
        unsupervised_data_with_rouge = generated_data_loading.get_generated_rouge(unsupervised_data, model,
                                                                                  search_params,
                                                                                  training_args.load_generated_model)
        return unsupervised_data_with_rouge.map(lambda example: {'rank': example['rouge-2-first']})
    if ranking == 'random':
        return unsupervised_data.map(lambda example: {'rank': random.random()})

    if ranking == 'filter' or ranking == 'ensemble':
        # write it all inline here, then extract components and unit test
        config = RankerConfig(
            num_summaries_per_text=1,

            ranker_learning_rate=1e-5,
            ranker_gradient_accumulation_steps=3,
            num_train_epochs=training_args.num_train_epochs,
            half_percision=False,
            do_evaluation=True,
            max_seq_len=0,

            loss_fn=training_args.ranker_loss_fn,
            tolerance=0.2,  # check it is ok, after I multiple by 100
            metric_for_best_model='accuracy_at_1',
            binary_classification=True,
            include_gold=True
        )

        assert train_dataset and validation_dataset
        # get filter and tokenizer by settings
        ranker_tokenizer = model_loading.get_ranker_tokenizer()

        train_dataset, validation_dataset = convert_to_regression_format(config, ranker_tokenizer, train_dataset,
                                                                         training_args,
                                                                         validation_dataset)

        unsupervised_data_for_ranking = processing.convert_generated_summaries_dataset_to_regression_dataset_format(
            unsupervised_data, ranker_tokenizer, max_num_summaries_per_text=config.num_summaries_per_text,
            max_seq_len=config.max_seq_len, binary_classification=True,
            include_gold=False, remove_text=False)

        if ranking == 'filter':
            ranker_model, ranker_tokenizer, trainer = train_ranker(config, train_dataset, validation_dataset)
            # results = trainer.predict(unsupervised_data_for_ranking)

            ranker_model.eval()
            unsupervised_data_special = unsupervised_data_for_ranking.map(
                lambda example: {'rank': ranker_model(**example)['logits'][0].item()})

            # if prediction_dataset:
            #     prediction_data_with_rouge = generated_data_loading.get_generated_rouge(predict_dataset, model,
            #                                                                             BeamSearchParams(num_beams=2,
            #                                                                                              num_return_sequences=2),
            #                                                                             training_args.load_generated_model)
            #
            #     prediction_data_with_rouge_as_labels = processing.convert_generated_summaries_dataset_to_regression_dataset_format(
            #         unsupervised_data, ranker_tokenizer,
            #         max_seq_len=config.max_seq_len, binary_classification=False,
            #         include_gold=False)
            #     trainer.evaluate(prediction_data_with_rouge_as_labels)

            return unsupervised_data_special

        if ranking == 'ensemble':
            unsupervised_data_for_ranking = unsupervised_data_for_ranking.map(lambda example: {'rank': 0.})
            k = 5
            for i in range(k):
                train_dataset = train_dataset.shuffle()
                validation_dataset = validation_dataset.shuffle()

                ranker_model, ranker_tokenizer, trainer = train_ranker(config, train_dataset.select(
                    range(int(max(1, 0.75 * len(train_dataset))))), validation_dataset.select(
                    range(int(max(1, 0.75 * len(validation_dataset))))))
                # results = trainer.predict(unsupervised_data_for_ranking)

                ranker_model.eval()
                unsupervised_data_for_ranking = unsupervised_data_for_ranking.map(
                    lambda example: {f'rank': example['rank'] + ranker_model(**example)['logits'][0].item()})
                compute.clean_memory()

            return unsupervised_data_for_ranking

    raise Exception('unknown ranking', ranking)


def train_ranker(config, train_dataset, validation_dataset):
    ranker_model, ranker_tokenizer = model_loading.get_ranker_model_and_tokenizer(config)
    # pass it train dataset(validation switch trick?) and validation dataset
    ranker_training_args = TrainingArguments(
        output_dir="./ranker_output_dir_" + str(time()).replace('.', '_'),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        do_train=True,
        overwrite_output_dir=True,
        # warmup_steps=0,
        fp16=config.half_percision,
        learning_rate=config.ranker_learning_rate,
        gradient_accumulation_steps=config.ranker_gradient_accumulation_steps,
        remove_unused_columns=False,
        evaluation_strategy='steps' if config.evaluate_every_steps else 'epoch' if config.do_evaluation else 'no',
        # load_best_model_at_end=True
        dataloader_num_workers=2,
        eval_steps=config.evaluate_every_steps,
        report_to=["comet_ml"],
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best_model,
        save_total_limit=1,
    )
    compute.clean_memory()
    trainer = training.train_ranker(ranker_model, config,
                                    ranker_training_args, train_dataset,
                                    eval_dataset=validation_dataset,
                                    test_dataset=None)
    return ranker_model, ranker_tokenizer, trainer


def convert_to_regression_format(config, ranker_tokenizer, train_dataset, training_args, validation_dataset):
    validation_dataset = processing.convert_generated_summaries_dataset_to_regression_dataset_format(
        validation_dataset, ranker_tokenizer, max_num_summaries_per_text=config.num_summaries_per_text,
        max_seq_len=config.max_seq_len,
        binary_classification=config.binary_classification, include_gold=config.include_gold)
    if training_args.train_filter_on == 'train' or training_args.train_filter_on == 'both':
        train_dataset = processing.convert_generated_summaries_dataset_to_regression_dataset_format(
            train_dataset, ranker_tokenizer, max_num_summaries_per_text=config.num_summaries_per_text,
            max_seq_len=config.max_seq_len, binary_classification=config.binary_classification,
            include_gold=config.include_gold)
    if training_args.train_filter_on == 'validation' or training_args.train_filter_on == 'both':
        splited = validation_dataset.train_test_split(train_size=len(train_dataset), shuffle=False)
        train_dataset2, validation_dataset = splited['train'], splited['test']
        # assert len(validation_dataset) >= 32
        if training_args.train_filter_on == 'both':
            train_dataset = datasets.concatenate_datasets([train_dataset, train_dataset2.map()])
            train_dataset.set_format('torch')
        else:
            train_dataset = train_dataset2
    return train_dataset, validation_dataset


def filter(ranked_dataset, amount_to_pass_filter=0.01):
    ranked_dataset = ranked_dataset.sort('rank', reverse=True)
    return ranked_dataset.select(range(max(1, int(amount_to_pass_filter * len(ranked_dataset)))))


ranked_unsupervised_dataset = rank(unsupervised_data, train_dataset, eval_dataset, training_args, predict_dataset)
filtered_unsupervised_dataset = filter(ranked_unsupervised_dataset, training_args.amount_to_pass_filter)
unsupervised_dataset_for_training = convert_dataset_with_generated_highlights_to_training_dataset(
    filtered_unsupervised_dataset, tokenizer, data_args)

# huggginface magic...
unsupervised_dataset_for_training.set_format(None)
do_train(model, tokenizer, unsupervised_dataset_for_training, eval_dataset, training_args, data_args, last_checkpoint)

final_rouge_on_test = my_eval(predict_dataset, model, tokenizer, search_params, description='on test set now')
log_metrics({'rouge2_on_test': final_rouge_on_test})
if rouge_on_test:
    log_metrics({'rouge-2-diff': final_rouge_on_test - rouge_on_test})

# generator model , generator tokenizer =


# train generator(train,validation)

# create summries and add rouge for train,validation, test

# train filter(train, validation)


# rank  unsupervised generated

# select top ranked

# insert into train set(can use some dataset feature to replace 'generated_highlight' for creating labels for generation trainig)
