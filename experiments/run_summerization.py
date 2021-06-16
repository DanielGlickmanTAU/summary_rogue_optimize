import comet_ml

from config.argument_parsing import DataTrainingArguments, GeneratorModelArguments
from data import data_loading
from models import model_loading
from utils import compute
import logging
import os
import sys

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_metric

from filelock import FileLock
from transformers import (
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.7.0.dev0")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt",
                   # paths=[compute.get_cache_dir()]
                   )
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True,
                      download_dir='/home/yandex/AMNLP2021/glickman1/anaconda3/envs/comet/nltk_data')


def run():
    data_args, model_args, training_args, last_checkpoint = parse_generation_args()

    model, tokenizer = model_loading.get_model_and_tokenizer(model_args)

    train_dataset, eval_dataset, predict_dataset = data_loading.get_dataset(data_args, training_args, tokenizer)

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    import random
    import time
    metric = load_metric("rouge", cache_dir=compute.get_cache_dir(), experiment_id=f'{random.random()}_{time.time()}')

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        print('evaluating in training')
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        result = {k: round(v, 4) for k, v in result.items()}
        return result

    label_smoothing_check(model, training_args)

    trainer = create_trainer(compute_metrics, data_collator, eval_dataset, model, tokenizer, train_dataset,
                             training_args)

    # Training
    if training_args.do_train:
        do_train(data_args, last_checkpoint, train_dataset, trainer, training_args)

    # Evaluation
    results = {}
    print('do eval', training_args.do_eval)
    if training_args.do_eval:
        do_eval(data_args, eval_dataset, trainer)

    if training_args.do_predict:
        do_predict(data_args, predict_dataset, tokenizer, trainer, training_args)

    return results


def parse_generation_args():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((GeneratorModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.cache_dir = compute.get_cache_dir()
    training_args.resume_from_checkpoint = None
    training_args.load_best_model_at_end = True
    training_args.fp16 = compute.get_torch().cuda.is_available()
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return data_args, model_args, training_args, last_checkpoint


def label_smoothing_check(model, training_args):
    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )


def do_predict(data_args, predict_dataset, tokenizer, trainer, training_args):
    logger.info("*** Predict ***")
    predict_results = trainer.predict(
        predict_dataset,
        metric_key_prefix="predict",
        max_length=data_args.val_max_target_length,
        num_beams=data_args.num_beams,
    )
    metrics = predict_results.metrics
    max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
    print('prediciton metrics', metrics)
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    log_metrics(metrics)

    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:
            predictions = tokenizer.batch_decode(
                predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            print(predictions)
            output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join(predictions))


def log_metrics(metrics):
    try:
        experiment = comet_ml.config.get_global_experiment()
        experiment.log_metrics(metrics)
    except:
        print('WARNING FAILED REPORTING METRICS', metrics)


def do_eval(data_args, eval_dataset, trainer):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(
        max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
    )
    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    print('eval metrics', metrics)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def create_trainer(compute_metrics, data_collator, eval_dataset, model, tokenizer, train_dataset, training_args):
    # Initialize our Trainer
    assert training_args.predict_with_generate
    assert training_args.do_eval and eval_dataset is not None

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )
    callback = EarlyStoppingCallback(early_stopping_patience=3)
    trainer.add_callback(callback)

    return trainer


def do_train(data_args, last_checkpoint, train_dataset, trainer, training_args):
    checkpoint = last_checkpoint
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    # elif last_checkpoint is not None:
    #     checkpoint = last_checkpoint
    if checkpoint:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
    else:
        train_result = trainer.train()

    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    run()
