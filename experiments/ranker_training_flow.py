from config.config import RankerConfig
from experiments import experiment
from utils import compute
import time
import gc

from transformers import TrainingArguments

from data import generated_data_loading
from models import model_loading as model_loading
from train import training


def run_exp(config: RankerConfig):
    tags = [f'num train examples {config.num_examples}', f'summaries per text {config.num_summaries_per_text}',
            config.train_mapped_saved_path,
            config.validation_mapped_saved_path,
            config.loss_fn]
    exp = experiment.start_experiment(hyperparams=vars(config), tags=tags)
    print(config)
    ranker_model, tokenizer = model_loading.get_ranker_model_and_tokenizer(config)
    binary_classification = False
    include_gold = True
    validation_processed_generated_xsum = generated_data_loading.load_processed_generated_dataset(
        config.validation_mapped_saved_path, config, tokenizer, max_examples=None,
        binary_classification=binary_classification,
        include_gold=include_gold)

    if config.test_mapped_saved_path:
        test_processed_generated_xsum = generated_data_loading.load_processed_generated_dataset(
            config.test_mapped_saved_path, config, tokenizer, max_examples=None,
            binary_classification=binary_classification,
            include_gold=include_gold
        )
    else:
        test_processed_generated_xsum = None

    train_processed_generated_xsum = generated_data_loading.load_processed_generated_dataset(
        config.train_mapped_saved_path, config, tokenizer, binary_classification=binary_classification,
        include_gold=include_gold)
    training_args = TrainingArguments(
        output_dir="./ranker_output_dir_" + str(time.time()).replace('.', '_'),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        do_train=True,
        overwrite_output_dir=False,
        # warmup_steps=0,
        fp16=config.half_percision,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
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

    training.train_ranker(ranker_model, config,
                          training_args, train_processed_generated_xsum,
                          eval_dataset=validation_processed_generated_xsum,
                          test_dataset=test_processed_generated_xsum)
