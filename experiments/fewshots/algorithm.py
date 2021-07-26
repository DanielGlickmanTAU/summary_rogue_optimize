import random

from utils import compute
from experiments.fewshots.learning import train_ranker
from experiments.fewshots.utils import convert_to_regression_format
from config.config import RankerConfig
from data import generated_data_loading, processing
from models import model_loading


def rank(model, unsupervised_data, train_dataset, validation_dataset, training_args, search_params):
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
            metric_for_best_model=training_args.ranker_metric_for_best_model,
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


def filter_dataset(ranked_dataset, amount_to_pass_filter=0.01):
    ranked_dataset = ranked_dataset.sort('rank', reverse=True)
    return ranked_dataset.select(range(max(1, int(amount_to_pass_filter * len(ranked_dataset)))))
