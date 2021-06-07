import datasets
import nltk
import numpy
from transformers import Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch

from data import metrics
from models.generation import add_summary_and_rouge

rouge = metrics.get_rouge()


def prepare_examples_for_training(examples, tokenizer):
    def assert_bart():
        assert 'Bart' in str(tokenizer.__class__)

    assert_bart()

    input_tokens = tokenizer(examples["article"], padding="max_length", truncation=True, max_length=512)
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

    for i in range(num_epochs):
        trainer = get_trainer(model, tokenizer, train_dataset, validation_dataset, batch_size,
                              learning_rate=learning_rate,
                              gradient_accumulation_steps=gradient_accumulation_steps, num_epochs=10)
        trainer.train()

        print(f'epoch {i}', trainer.evaluate(
            max_length=128, num_beams=4, metric_key_prefix="eval"
        ))
        new_valid_score = eval_metric(model, tokenizer, validation_dataset, exp, search_params)
        print(f'rouge 2 on validation in iteration {i} is {new_valid_score}')


# trains generaiton
def get_trainer(model, tokenizer, train_dataset, eval_dataset, batch_size, learning_rate,
                gradient_accumulation_steps=1,
                num_epochs=1):
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metric(eval_preds):
        preds, labels = eval_preds
        preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds.argmax(2), skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = numpy.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [numpy.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = numpy.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # def compute_metric(pred, **args):
    #     # print('compute metric a1', pred)
    #     print('compute metric args', args)
    #     labels_ids = pred.label_ids
    #     pred_ids = pred.predictions
    #     if isinstance(pred_ids, tuple):
    #         pred_ids = pred.predictions[0]
    #
    #     loss = torch.nn.CrossEntropyLoss()(torch.tensor(pred_ids[0]).squeeze(0), torch.tensor(labels_ids).squeeze(0))
    #     pred_str = tokenizer.batch_decode(pred_ids.argmax(2), skip_special_tokens=True, )
    #     labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    #     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    #
    #     rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
    #     print(rouge_output)
    #     print('predictd str:', pred_str)
    #     print('label_str:', label_str)
    #     return {'loss': loss, 'rouge2': rouge_output.fmeasure}

    train_dataset = prepare_split_for_training(train_dataset, tokenizer, batch_size)
    eval_dataset = prepare_split_for_training(eval_dataset, tokenizer, batch_size)

    training_args = Seq2SeqTrainingArguments(
        # predict_with_generate=True,
        output_dir="./",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        do_train=True,
        do_eval=True,
        evaluation_strategy='epoch',
        # prediction_loss_only=True,

        overwrite_output_dir=False,
        # warmup_steps=0,
        fp16=torch.cuda.is_available(),
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metric,

    )

    return trainer
