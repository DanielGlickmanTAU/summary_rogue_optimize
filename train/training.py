from typing import Dict

from transformers import TrainingArguments, EarlyStoppingCallback
import torch

from evaluation.evaluate import best_at_k
from train.RankerTrainer import RankerTrainer


# from transformers.modeling_bart import shift_tokens_right


def ranker_data_collator(features) -> Dict[str, torch.Tensor]:
    # features is a list of size #batch_size
    # each item in it is a dict with  keys attention_mask_s,input_ids_s,labels. each key value is a list with size #num_beams.
    # length of labels is also num_beams. length of attention_mask_s and input_ids_s is tokenizor length
    # print(features)
    # lets assume batch_size is 1 for now
    batch_size = len(features)
    assert batch_size == 1, f'{features} not with size 1'
    features_0 = features[0]
    input_ids_s = torch.stack(features_0['input_ids_s'])
    attention_mask_s = torch.stack(features_0['attention_mask_s'])
    return {
        'input_ids_s': input_ids_s,
        'attention_mask_s': attention_mask_s,
        'labels': features_0['labels'].float()
    }


done_oracle = set()


def train_ranker(ranker_model, config, training_arguments: TrainingArguments, dataset, eval_dataset=None,
                 test_dataset=None):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions, labels = torch.tensor(predictions), torch.tensor(labels)
        d = {}

        is_gold_in_labels = bool(labels[0:, 0].mean() == 1)
        labels_hash = (labels[0] + labels[-1]).sum().item()

        if is_gold_in_labels:
            for k in range(1, labels.shape[-1]):
                accuracy_k = (predictions[:, 0:k + 1].argmax(dim=1) == 0).float().mean().item()
                d[f'accuracy_at_{k}'] = accuracy_k
            labels = labels[:, 1:]
            predictions = predictions[:, 1:]
        if labels_hash not in done_oracle:
            done_oracle.add(labels_hash)
            for k in range(1, labels.shape[-1] + 1):
                oracle_at_k, average_at_k = best_at_k(labels, labels, k)
                print(f'oracle rouge best and average at {k}:', oracle_at_k, average_at_k, is_gold_in_labels)
                d[f'oracle_at_{k}'] = oracle_at_k
                d[f'average_at_{k}'] = average_at_k
        for k in range(1, labels.shape[-1] + 1):
            selected_at_k, average_at_k = best_at_k(labels, predictions, k)
            d[f'selected_at_{k}'] = selected_at_k

        print(d)
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

    callback = EarlyStoppingCallback(early_stopping_patience=5)
    trainer.add_callback(callback)

    trainer.train()

    if test_dataset:
        print('STARTING PREDICT')
        predict_results = trainer.predict(
            test_dataset,
            metric_key_prefix="predict",
        )
        print(predict_results.metrics)
    return trainer
