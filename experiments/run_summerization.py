from time import time

import comet_ml

from config.argument_parsing import parse_generation_args
from data import data_loading, metrics
from data.metrics import postprocess_text, compute_rouge_from_token_ids
from evaluation import evaluate
from models import model_loading, generation
import logging
import os

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np

from filelock import FileLock
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from transformers.file_utils import is_offline_mode

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.7.0.dev0")
from models.generate import BeamSearchParams
from train import generation_training

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

    trainer = generation_training.create_trainer(train_dataset, eval_dataset, training_args, data_args, model,
                                                 tokenizer)

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


def do_predict(data_args, predict_dataset, tokenizer, trainer, training_args):
    logger.info("*** Predict ***")
    print('predict item firt', predict_dataset[0])
    ds = predict_dataset.map(lambda x: generation.add_summary_and_rouge(trainer.model, tokenizer, x,
                                                                        BeamSearchParams(num_return_sequences=1,
                                                                                         num_beams=data_args.num_beams)),
                             batched=True, batch_size=4)

    evaluate.print_rouge_stuff(ds)
    print('done shit')
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
            for i in range(len(predictions)):
                mine = ds[i]['generated_highlights'][0]
                print('mine', mine)
                theirs = predictions[i]
                print('theirs', theirs)
                assert mine == theirs
                # output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                # with open(output_prediction_file, "w") as writer:
                #     writer.write("\n".join(predictions))


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

    if training_args.save_model_after_train:
        t = time()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        print(f'saving took {time() - t} seconds')
    else:
        print('skiping saving generation model')
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    run()
