from utils import compute
from experiments.fewshots.learning import train_ranker, do_evaluate, do_train
from experiments.fewshots.utils import convert_to_regression_format

from config.argument_parsing import parse_generation_args
from config.config import RankerConfig
from data import data_loading, generated_data_loading, processing
from data.processing import convert_dataset_with_generated_highlights_to_training_dataset
from experiments import experiment
from experiments.experiment import log_metrics
from models import model_loading, checkpoints
import random
import os

from models.generate import BeamSearchParams

data_args, model_args, training_args, last_checkpoint = parse_generation_args()

search_params = BeamSearchParams(num_return_sequences=1, num_beams=data_args.num_beams)

model_checkpoint = \
    checkpoints.get_checkpoint_output_dir(data_args.dataset_name, model_args.model_name_or_path,
                                          data_args.max_train_samples, training_args.learning_rate,
                                          extra=training_args.shuffle_seed if training_args.shuffle_training_set else None)
training_args.output_dir = model_checkpoint + str(random.random())

experiment.start_experiment(hyperparams=[data_args, training_args, model_args],
                            tags=[] if training_args.track_experiment else [
                                'debug'])

original_model_name_or_path = model_args.model_name_or_path

tokenizer = model_loading.get_generator_tokenizer(model_args)

train_dataset, eval_dataset, predict_dataset, unsupervised_data = data_loading.get_dataset(data_args, training_args,
                                                                                           tokenizer,
                                                                                           do_unsupervised=True)

if training_args.load_generated_model and os.path.isdir(model_checkpoint):

    model_args.model_name_or_path = model_checkpoint
    model, tokenizer = model_loading.get_model_and_tokenizer(model_args)
else:
    model, tokenizer = model_loading.get_model_and_tokenizer(model_args)
    model_save_path = model_checkpoint if training_args.load_generated_model else None
    do_train(model, tokenizer, train_dataset, eval_dataset, training_args, data_args, last_checkpoint,
             model_name_or_path_for_saving=model_save_path)

# eval on test test
rouge_on_test = None
if not training_args.skip_first_test_eval:
    rouge_on_test = do_evaluate(predict_dataset, model, tokenizer, search_params,
                                f'on TEST set after training on {len(train_dataset)} samples')
    log_metrics({'rouge2_on_test': rouge_on_test})
    log_metrics({'rouge2_on_test_first': rouge_on_test})

# here do if training_args.use_gpt_dataset load from my new python file... else below
if not training_args.use_gpt_dataset:
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


def rank(unsupervised_data, train_dataset, validation_dataset, training_args):
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

            ranker_learning_rate=training_args.ranker_learning_rate,
            ranker_gradient_accumulation_steps=training_args.ranker_gradient_accumulation_steps,
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


def filter(ranked_dataset, amount_to_pass_filter=0.01):
    ranked_dataset = ranked_dataset.sort('rank', reverse=True)
    return ranked_dataset.select(range(max(1, int(amount_to_pass_filter * len(ranked_dataset)))))


ranked_unsupervised_dataset = rank(unsupervised_data, train_dataset, eval_dataset, training_args)
filtered_unsupervised_dataset = filter(ranked_unsupervised_dataset, training_args.amount_to_pass_filter)
unsupervised_dataset_for_training = convert_dataset_with_generated_highlights_to_training_dataset(
    filtered_unsupervised_dataset, tokenizer, data_args)

# huggginface magic...
unsupervised_dataset_for_training.set_format(None)
if training_args.train_from_scratch_on_unsupervised:
    del (model)
    compute.clean_memory()

    model_args.model_name_or_path = original_model_name_or_path
    model, tokenizer = model_loading.get_model_and_tokenizer(model_args)
    do_train(model, tokenizer, unsupervised_dataset_for_training, eval_dataset, training_args, data_args,
             last_checkpoint, model_name_or_path_for_saving=None)
    do_train(model, tokenizer, train_dataset, eval_dataset, training_args, data_args, last_checkpoint,
             model_name_or_path_for_saving=None)
else:
    do_train(model, tokenizer, unsupervised_dataset_for_training, eval_dataset, training_args, data_args,
             last_checkpoint, model_name_or_path_for_saving=None)

final_rouge_on_test = do_evaluate(predict_dataset, model, tokenizer, search_params, description='on test set now')
log_metrics({'rouge2_on_test': final_rouge_on_test})
log_metrics({'rouge2_on_test_last': final_rouge_on_test})
if rouge_on_test:
    log_metrics({'rouge-2-diff': final_rouge_on_test - rouge_on_test})
