from config.argument_parsing import parse_generation_args
from data import data_loading
from evaluation import evaluate
from models import model_loading, generation
from time import time

# TODO this also exists in run_summarization, origanize and move this to one place
from models.generate import BeamSearchParams
from train import generation_training
from utils import decorators


def do_eval(data_args, eval_dataset, trainer):
    t0 = time()
    print('evaluating')
    metrics = trainer.evaluate(
        max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
    )
    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    print('eval metrics', metrics)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    print(f'do_eval took {time() - t0}')


@decorators.measure_time
def my_eval(dataset, model, tokenizer):
    ds = generation.add_summary_and_rouge(model, tokenizer, dataset,
                                          BeamSearchParams(num_return_sequences=1,
                                                           num_beams=data_args.num_beams))
    return evaluate.print_rouge_stuff(ds)


# unsupervised_data = unsupervised_data.select(range(128))


# eval_backup = eval_dataset.map()


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

model, tokenizer = model_loading.get_model_and_tokenizer(model_args)

train_dataset, eval_dataset, predict_dataset, unsupervised_data = data_loading.get_dataset(data_args, training_args,
                                                                                           tokenizer,
                                                                                           do_unsupervised=True)

my_eval(eval_dataset, model, tokenizer)
do_train(model, tokenizer, train_dataset, eval_dataset, training_args, data_args, last_checkpoint)

my_eval(eval_dataset, model, tokenizer)

# add rouge on unsupervised_data <-- need this only for top scoring rouge baseline

# rank(unsupervised_data)
# generator model , generator tokenizer =


# train generator(train,validation)

# create summries and add rouge for train,validation, test

# train filter(train, validation)


# rank  unsupervised generated

# select top ranked

# insert into train set(can use some dataset feature to replace 'generated_highlight' for creating labels for generation trainig)
