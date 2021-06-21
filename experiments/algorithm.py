from config.argument_parsing import parse_generation_args
from data import data_loading
from evaluation import evaluate
from models import model_loading, generation, checkpoints
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
def my_eval(dataset, model, tokenizer, search_params):
    ds = generation.add_summary_and_rouge(model, tokenizer, dataset,
                                          search_params)
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
search_params = BeamSearchParams(num_return_sequences=1, num_beams=data_args.num_beams)

if training_args.load_generated_model:
    model_checkpoint = \
        checkpoints.get_checkpoint_output_dir(data_args.dataset_name, model_args.model_name_or_path,
                                              data_args.max_train_samples, training_args.learning_rate, extra=None)
    model_args.model_name_or_path = model_checkpoint
    model, tokenizer = model_loading.get_model_and_tokenizer(model_args)
else:
    model, tokenizer = model_loading.get_model_and_tokenizer(model_args)

train_dataset, eval_dataset, predict_dataset, unsupervised_data = data_loading.get_dataset(data_args, training_args,
                                                                                           tokenizer,
                                                                                           do_unsupervised=True)

do_train(model, tokenizer, train_dataset, eval_dataset, training_args, data_args, last_checkpoint)
my_eval(eval_dataset, model, tokenizer, search_params)

unsupervised_data = generation.add_summary(model, tokenizer, unsupervised_data, search_params,
                                           batch_size=training_args.per_device_eval_batch_size)


# add rouge on unsupervised_data <-- need this only for top scoring rouge baseline

def rank(unsupervised_data):
    unsupervised_data_with_rouge = generation.add_rouge(unsupervised_data)
    return unsupervised_data_with_rouge.map(lambda example: {'rank': example['rouge-2-first']})


def filter(ranked_dataset):
    ranked_dataset = ranked_dataset.sort('rank', reverse=True)
    return ranked_dataset.select(range(int(0.01 * len(ranked_dataset))))


def convert_dataset_with_generated_highlights_to_training_dataset(dataset):
    return dataset.map(
        lambda example: {'highlights': example['generated_highlights'][0]}
    )


ranked_unsupervised_dataset = rank(unsupervised_data)
filtered_unsupervised_dataset = filter(ranked_unsupervised_dataset)
unsupervised_dataset_for_training = convert_dataset_with_generated_highlights_to_training_dataset(
    filtered_unsupervised_dataset)

do_train(model, tokenizer, unsupervised_dataset_for_training, eval_dataset, training_args, data_args, last_checkpoint)
my_eval(eval_dataset, model, tokenizer, search_params)

# generator model , generator tokenizer =


# train generator(train,validation)

# create summries and add rouge for train,validation, test

# train filter(train, validation)


# rank  unsupervised generated

# select top ranked

# insert into train set(can use some dataset feature to replace 'generated_highlight' for creating labels for generation trainig)
