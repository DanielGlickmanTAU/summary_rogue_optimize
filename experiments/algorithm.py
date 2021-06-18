from config.argument_parsing import parse_generation_args
from data import data_loading
from evaluation import evaluate
from models import model_loading, generation
from time import time

# TODO this also exists in run_summarization, origanize and move this to one place
from models.generate import BeamSearchParams
from train import generation_training


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
        trainer.save_state()
        print(f'saving took {time() - t} seconds')
    else:
        print('skiping saving generation model')
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)


data_args, model_args, training_args, last_checkpoint = parse_generation_args()

model, tokenizer = model_loading.get_model_and_tokenizer(model_args)

train_dataset, eval_dataset, predict_dataset, unsupervised_data = data_loading.get_dataset(data_args, training_args,
                                                                                           tokenizer,
                                                                                           do_unsupervised=True)
# unsupervised_data = unsupervised_data.select(range(128))
# training_args.remove_unused_columns = False
# eval_backup = eval_dataset.map()

trainer = generation_training.create_trainer(unsupervised_data, eval_dataset, training_args, data_args, model,
                                             tokenizer)

do_train(data_args, last_checkpoint, train_dataset, trainer, training_args)

print('after training')
do_eval(data_args, eval_dataset, trainer)
ds = eval_backup.map(lambda x: generation.add_summary_and_rouge(trainer.model, tokenizer, x,
                                                                BeamSearchParams(num_return_sequences=2,
                                                                                 num_beams=data_args.num_beams)),
                     batched=True, batch_size=4)

t0 = time()
evaluate.print_rouge_stuff(ds)
print(f'my add_summary and rouge took {time() - t0}')
# generator model , generator tokenizer =


# train generator(train,validation)

# create summries and add rouge for train,validation, test

# train filter(train, validation)


# rank  unsupervised generated

# select top ranked

# insert into train set(can use some dataset feature to replace 'generated_highlight' for creating labels for generation trainig)
