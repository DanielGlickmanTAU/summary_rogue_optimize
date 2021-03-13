from data import cnn_dataset, metrics
from models import model_loading, generate
from models.candidate_selection import select_best
from transformers import Trainer, TrainingArguments

batch_size = 16
train_examples = 16 * 5
validation_examples = 16 * 1

model, tokenizer = model_loading.get_bart_model_and_tokenizer()
cnn = cnn_dataset.get_cnn_dataset(train_subset=train_examples, valid_subset=validation_examples)
rouge = metrics.get_rouge()


def add_summary_and_rouge(examples):
    articles = examples['article']
    gold = examples['highlights']
    generated_summaries = generate.summarize(model, tokenizer, articles)

    assert len(gold) == len(generated_summaries)
    scores = [metrics.calc_score(pred, ref) for pred, ref in zip(generated_summaries, gold)]
    rouge2 = [x['rouge-2'] for x in scores]
    rouge1 = [x['rouge-1'] for x in scores]

    return {'generated_summaries': generated_summaries, 'rouge2': rouge2, 'rouge1': rouge1}


def eval_metric(dataset_split):
    ds = dataset_split.map(add_summary_and_rouge, batched=True, batch_size=batch_size, keep_in_memory=True)
    ds_rouge_ = sum(ds['rouge2']) / len(ds['rouge2'])
    print('rouge2 is ', ds_rouge_, ' evaluate on', len(ds['rouge2']))
    return ds_rouge_


def prepare_examples_for_training(examples, tokenizer):
    highlight_tokens = tokenizer(examples["highlights"], padding="max_length", truncation=True, max_length=128)

    decoder_input_ids = highlight_tokens['input_ids']
    decoder_attention_mask = highlight_tokens['attention_mask']
    labels = highlight_tokens['input_ids'].copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in tokens] for tokens in labels]

    return {'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels
            }


def prepare_split_for_training(train_data, tokenizer):
    train_data = train_data.map(
        lambda examples: prepare_examples_for_training(examples, tokenizer),
        batched=True,
        batch_size=batch_size,
        # todo consider here removing 'generated_summary' field
        remove_columns=["article", "highlights", "id"]
    )
    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )
    return train_data


def train(model, tokenizer, mini_split):
    mini_split = prepare_split_for_training(mini_split, tokenizer)

    training_args = TrainingArguments(
        # output_dir="./",
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # predict_from_generate=True,
        # evaluate_during_training=True,
        do_train=True,
        do_eval=False,
        # logging_steps=1000,
        # save_steps=1000,
        # eval_steps=1000,
        overwrite_output_dir=False,
        # warmup_steps=2000,
        # save_total_limit=3,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        # compute_metrics=compute_metrics,
        train_dataset=mini_split,
        # eval_dataset=val_dataset,
    )

    trainer.train()


test_summaries = cnn['test'].map(add_summary_and_rouge, batched=True, batch_size=batch_size)
current_valid_score = eval_metric(cnn['validation'])
while True:
    top = select_best(test_summaries)
    # replace gold tags with generated
    # comment this out when I want to compare to normal training.. and also set select scale_exp=0
    top = top.map(lambda examples: {'highlights': examples['generated_summaries']})
    train(model, tokenizer, top)

    new_valid_score = eval_metric(cnn['validation'])
    if new_valid_score <= current_valid_score:
        break
    current_valid_score = new_valid_score
    test_summaries = cnn['test'].map(add_summary_and_rouge, batched=True, batch_size=batch_size)

print('rouge2', sum(test_summaries['rouge2']) / len(test_summaries['rouge2']))
print('rouge1', sum(test_summaries['rouge1']) / len(test_summaries['rouge1']))

print('rouge2 top', sum(top['rouge2']) / len(top['rouge2']))
print('rouge1 top', sum(top['rouge1']) / len(top['rouge1']))
