from typing import Dict

from transformers import TrainingArguments, Trainer, Seq2SeqTrainingArguments
import torch
from transformers.models.bart.modeling_bart import shift_tokens_right

from evaluation.evaluate import best_at_k
from train.RankerTrainer import RankerTrainer

# from transformers.modeling_bart import shift_tokens_right

learning_rate = 3e-06


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
    # labels = [[-100 if token == tokenizer.pad_token_id else token for token in tokens] for tokens in labels]

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


def ranker_data_collator(features) -> Dict[str, torch.Tensor]:
    # features is a list of size #batch_size
    # each item in it is a dict with  keys attention_mask_s,input_ids_s,labels. each key value is a list with size #num_beams.
    # length of labels is also num_beams. length of attention_mask_s and input_ids_s is tokenizor length
    # print(features)
    # lets assume batch_size is 1 for now
    batch_size = len(features)
    assert batch_size == 1
    features_0 = features[0]
    input_ids_s = torch.stack(features_0['input_ids_s'])
    attention_mask_s = torch.stack(features_0['attention_mask_s'])
    return {
        'input_ids_s': input_ids_s,
        'attention_mask_s': attention_mask_s,
        'labels': features_0['labels'].float()
    }


done_oracle = False


def train_ranker(ranker_model, config, training_arguments: TrainingArguments, dataset, eval_dataset=None):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions, labels = torch.tensor(predictions), torch.tensor(labels)
        d = {}

        global done_oracle
        if not done_oracle:
            done_oracle = True
            for k in range(1, labels.shape[-1] + 1):
                oracle_at_k, average_at_k = best_at_k(labels, labels, k)
                print(f'oracle rouge best and average at {k}:', oracle_at_k, average_at_k)
                d[f'oracle_at_{k}'] = oracle_at_k
                d[f'average_at_{k}'] = average_at_k
        for k in range(1, labels.shape[-1] + 1):
            selected_at_k, average_at_k = best_at_k(labels, predictions, k)
            print(f'eval rouge best and average at {k}:', selected_at_k, average_at_k)
            d[f'selected_at_{k}'] = selected_at_k
            print('\n' * 5)

        return d

    assert not training_arguments.remove_unused_columns

    trainer = RankerTrainer(
        model=ranker_model,
        args=training_arguments,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=ranker_data_collator,
        compute_metrics=compute_metrics,
        config=config
    )

    trainer.train()


# trains generaiton
def train(model, tokenizer, train_dataset, eval_dataset, batch_size, learning_rate=learning_rate,
          gradient_accumulation_steps=1,
          num_epochs=1):
    # def compute_metric(a1, **args):
    #     print('compute metric a1', a1)
    #     print('compute metric args', args)
    #     return {}

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
        prediction_loss_only=True,

        overwrite_output_dir=False,
        # warmup_steps=0,
        fp16=torch.cuda.is_available(),
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # compute_metrics=compute_metric,

    )

    trainer.train()
