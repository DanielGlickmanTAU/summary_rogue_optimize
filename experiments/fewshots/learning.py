from utils import compute, decorators
import os
from time import time

from transformers import TrainingArguments

from evaluation import evaluate
from models import model_loading, generation
from train import training, generation_training


def train_ranker(config, train_dataset, validation_dataset):
    ranker_model, ranker_tokenizer = model_loading.get_ranker_model_and_tokenizer(config)
    # pass it train dataset(validation switch trick?) and validation dataset
    output_dir = "./ranker_output_dir_" + str(time()).replace('.', '_')
    ranker_training_args = TrainingArguments(
        output_dir=output_dir,
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
        save_strategy='no'
    )
    compute.clean_memory()
    trainer = training.train_ranker(ranker_model, config,
                                    ranker_training_args, train_dataset,
                                    eval_dataset=validation_dataset,
                                    test_dataset=None)

    os.system(f'rm -rf {output_dir}')

    return ranker_model, ranker_tokenizer, trainer


@decorators.measure_time
def do_evaluate(dataset, model, tokenizer, search_params, description=''):
    ds = generation.add_summary_and_rouge(model, tokenizer, dataset,
                                          search_params)
    print(f'evaluate {description}')
    return evaluate.print_rouge_stuff(ds)


@decorators.measure_time
def do_train(model, tokenizer, train_dataset, eval_dataset, training_args, data_args, last_checkpoint,
             model_name_or_path_for_saving=None):
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

    if training_args.load_generated_model and model_name_or_path_for_saving:
        t = time()
        trainer.save_model(model_name_or_path_for_saving)  # Saves the tokenizer too for easy upload
        trainer.save_state()
        print(f'saving took {time() - t} seconds')
    else:
        print('skiping saving generation model')
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)

    # delete checkpoints or else it willl fill up the disk
    os.system(f'rm -rf {training_args.output_dir}')
    return metrics
