from typing import Dict

from transformers import TrainingArguments, Trainer
import torch

from train.RankerTrainer import RankerTrainer

learning_rate = 1e-05


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

    # return features


done_oracle = False


def train_ranker(ranker_model, tokenizer, training_arguments: TrainingArguments, dataset,
                 eval_dataset=None):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions, labels = torch.tensor(predictions), torch.tensor(labels)

        mx = predictions.argmax(dim=1)
        max_selected = labels[torch.arange(labels.shape[0]), mx]
        total = max_selected.mean()

        global done_oracle
        if not done_oracle:
            done_oracle = True
            for k in range(1, labels.shape[-1] + 1):
                print(f'oracle rouge best and average at {k}:', best_at_k(labels, labels, k))
        # print('compute_metrics predictions', predictions)
        # print('compute_metrics labels', labels)
        # print('best indexes per sample', mx)
        # print('corrspond to real rouge', max_selected)
        for k in range(1, labels.shape[-1] + 1):
            print(f'eval rouge best and average at {k}:', best_at_k(labels, predictions, k))
        print('\n' * 5)

        return {'eval_loss': total}

    def best_at_k(labels_tensor, index_tensor, k=None):
        if not k:
            k = labels_tensor.shape[0]
        best_indexes = index_tensor[:, 0:k].argmax(dim=1)
        labels_value_at_index = labels_tensor[torch.arange(labels_tensor.shape[0]), best_indexes]
        average_at_k = labels_tensor[torch.arange(labels_tensor.shape[0]), 0:k].mean().item()
        # print(index_tensor)
        # print(labels_tensor)
        # print('best indexes', best_indexes)
        # print('values choosen:', labels_value_at_index)
        return labels_value_at_index.mean().item(), average_at_k

    assert training_arguments.remove_unused_columns == False

    trainer = RankerTrainer(
        model=ranker_model,
        args=training_arguments,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=ranker_data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # trains generaiton for single epoch
    def train(model, tokenizer, mini_split, batch_size, learning_rate=learning_rate, gradient_accumulation_steps=1):
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
            fp16=True,
            prediction_loss_only=True,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=mini_split,
        )

        trainer.train()
