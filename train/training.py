from typing import Dict

from torch.cuda.amp import autocast
from transformers import TrainingArguments, Trainer, EvalPrediction
import torch
from transformers.trainer_pt_utils import nested_detach, DistributedTensorGatherer, nested_concat

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


def train_ranker(ranker_model, tokenizer, training_arguments: TrainingArguments, dataset,
                 eval_dataset=None):
    def compute_metrics(eval_pred):
        for i in range(1000):
            print('waitttt')
        predictions, labels = eval_pred
        return {'bla': 2.}

    assert training_arguments.remove_unused_columns == False

    trainer = MyTrainer(
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


class MyTrainer(Trainer):
    def __init__(self, **kwargs):
        super(MyTrainer, self).__init__(**kwargs)

    # def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys, ):
    #     loss, logits, labels = super(MyTrainer, self).prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    #     print('loss', 'logits', 'labels')
    #     print(loss, logits, labels)
    #     # if eval return best? label[argmax(logits)]
    #     print('would choose', labels[logits.argmax()])
    #     if not model.training:
    #         return loss, labels[logits.argmax()], labels
    #     return loss, logits, labels

    def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only,
            ignore_keys):
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if isinstance(outputs, dict):
                    loss = outputs["loss"].mean().detach()
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    loss = outputs[0].mean().detach()
                    logits = outputs[1:]
            else:
                loss = None
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        print('in prediction step', logits, labels)
        return (loss, logits, labels)

    def prediction_loop(
            self,
            dataloader,
            description,
            prediction_loss_only=None,
            ignore_keys=None,
            metric_key_prefix="eval",
    ):
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1 and not self.args.model_parallel:
            model = torch.nn.DataParallel(model)
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)

        losses_host: torch.Tensor = None
        labels_host = None
        preds_host = None
        labels_hos = None

        world_size = 1

        if self.args.local_rank != -1:
            world_size = torch.distributed.get_world_size()
        world_size = max(1, world_size)

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples)

        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            if loss is not None:
                print('loss before repeat', loss.shape)
                losses = loss.repeat(batch_size)
                print('loss after repeat', loss.shape)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)

            if logits is not None:
                if preds_host is not None:
                    print(preds_host, ' is not none something is wrong')
                logits = logits.unsqueeze(0)
                print('logits is ', logits.shape)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = labels.unsqueeze(0)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                    labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        if not prediction_loss_only:
            preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    # def compute_loss(self, model, inputs):
    #     print('yoyo')
    #     """
    #     How the loss is computed by Trainer. By default, all models return the loss in the first element.
    #
    #     Subclass and override for custom behavior.
    #     """
    #     print('first thing first', inputs)
    #     outputs = model(**inputs)
    #     # Save past state if it exists
    #     # TODO: this needs to be fixed and made cleaner later.
    #     if self.args.past_index >= 0:
    #         self._past = outputs[self.args.past_index]
    #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
    #     print('this how we do', outputs)
    #     return outputs["loss"] if isinstance(outputs, dict) else outputs[0]
