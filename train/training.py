from transformers import TrainingArguments, Trainer


def prepare_examples_for_training(examples, tokenizer):
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
        # todo consider here removing 'generated_summary' field
        remove_columns=["article", "highlights", "id"]
    )
    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )
    return train_data


def train(model, tokenizer, mini_split, batch_size):
    mini_split = prepare_split_for_training(mini_split, tokenizer, batch_size)

    training_args = TrainingArguments(
        output_dir="./",
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        do_train=True,
        do_eval=False,
        overwrite_output_dir=False,
        # warmup_steps=0,
        # fp16=True,
        prediction_loss_only=True,
        learning_rate=4e-05
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mini_split,
    )

    trainer.train()
