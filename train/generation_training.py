import experiments.fewshots.learning
from utils import compute, decorators
from train.FixedCometCallback import FixedCometCallback
import datasets
import nltk
import transformers

from config.consts import bert_max_len
from data.metrics import compute_rouge_from_token_ids
import numpy
from transformers import Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, \
    DataCollatorForSeq2Seq
import torch

from data import metrics
from models.generation import add_summary_and_rouge

rouge = metrics.get_rouge()


def prepare_examples_for_training(examples, tokenizer):
    def assert_bart():
        assert 'Bart' in str(tokenizer.__class__)

    assert_bart()

    input_tokens = tokenizer(examples["article"], padding="max_length", truncation=True, max_length=bert_max_len)
    highlight_tokens = tokenizer(examples["highlights"], padding="max_length", truncation=True, max_length=128)

    decoder_input_ids = highlight_tokens['input_ids']
    decoder_attention_mask = highlight_tokens['attention_mask']
    labels = highlight_tokens['input_ids'].copy()
    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in tokens] for tokens in labels]

    return {
        'input_ids': input_tokens['input_ids'],
        'attention_mask': input_tokens['attention_mask'],
        'decoder_input_ids': decoder_input_ids,
        'decoder_attention_mask': decoder_attention_mask,
        'labels': labels
    }


def prepare_split_for_training(train_data, tokenizer, batch_size):
    train_data = train_data.map(
        lambda examples: prepare_examples_for_training(examples, tokenizer),
        batched=True,
        batch_size=batch_size,
        remove_columns=["article", "highlights", "id"] if 'id' in train_data else ["article", "highlights"]
    )
    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )
    return train_data


@decorators.measure_time
def create_trainer(train_dataset, eval_dataset, training_args, data_args, model, tokenizer):
    def label_smoothing_check(model, training_args):
        if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
            print(
                "WARNING label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )

    # Initialize our Trainer
    assert training_args.predict_with_generate
    assert training_args.do_eval and eval_dataset is not None
    label_smoothing_check(model, training_args)

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        return compute_rouge_from_token_ids(preds, labels, tokenizer, data_args.ignore_pad_token_for_loss)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if experiments.fewshots.learning.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,

        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )
    callback = EarlyStoppingCallback(early_stopping_patience=3)
    trainer.add_callback(callback)
    try:
        trainer.pop_callback(transformers.integrations.CometCallback)
    except:
        pass

    trainer.callback_handler.add_callback(FixedCometCallback(training_args))

    return trainer


def generation_train_flow(model, tokenizer, exp, search_params, train_dataset, validation_dataset, batch_size,
                          learning_rate, gradient_accumulation_steps, num_epochs):
    def eval_metric(model, tokenizer, dataset_split, exp, search_params):

        ds = dataset_split.map(lambda x: add_summary_and_rouge(model, tokenizer, x, search_params),
                               batched=True, batch_size=4)
        print(ds[0]['generated_highlights'])
        ds_rouge_2 = sum(ds['rouge-2-first']) / len(ds['rouge-2-first'])
        ds_rouge_avg = sum(ds['rouge-2-avg']) / len(ds['rouge-2-avg'])
        # ds_rouge_1 = sum(ds['rouge1']) / len(ds['rouge1'])
        print('rouge2 when selecting first beam is ', ds_rouge_2,
              'rouge2 averaging ', search_params.num_beams, ' is ', ds_rouge_avg,
              ' evaluated on', len(ds['rouge-2-first']))
        # print('rouge1 is ', ds_rouge_1, ' evaluate on', len(ds['rouge2']))
        try:
            exp.log_metrics({'rouge2': ds_rouge_2})
        except Exception:
            pass
        return ds_rouge_2

    train_dataset = prepare_split_for_training(train_dataset, tokenizer, batch_size)
    validation_dataset = prepare_split_for_training(validation_dataset, tokenizer, batch_size)

    for i in range(num_epochs):
        # this will fail... switch from old trainer to new trainer
        trainer = create_trainer(model, tokenizer, train_dataset, validation_dataset, batch_size,
                                 learning_rate=learning_rate,
                                 gradient_accumulation_steps=gradient_accumulation_steps, num_epochs=100)
        trainer.train()

        print(f'epoch {i}', trainer.evaluate(
            max_length=128, num_beams=4, metric_key_prefix="eval"
        ))
        predict(tokenizer, trainer, validation_dataset, search_params)


def predict(tokenizer, trainer, test_dataset, search_params):
    predict_results = trainer.predict(
        test_dataset,
        metric_key_prefix="predict",
        max_length=128,
        num_beams=search_params.num_beams,
        # num_return_sequences=search_params.num_return_sequences
    )
    metrics = predict_results.metrics

    metrics["predict_samples"] = len(test_dataset)
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    assert trainer.args.predict_with_generate
    if trainer.is_world_process_zero():
        predictions = tokenizer.batch_decode(
            predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        predictions = [pred.strip() for pred in predictions]
        print('predictions:', predictions)
