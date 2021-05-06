from utils import compute

compute.get_torch()
from transformers import BatchEncoding, TrainingArguments

from config.config import RankerConfig
from data import generated_data_loading, processing
from models import model_loading
from train import training
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchtest import assert_vars_change

# what are the variables?
# print('Our list of parameters', [np[0] for np in model.named_parameters()])
config = RankerConfig(
    num_examples=2,
    num_skip=2,
    num_beams=2,
    learning_rate=1e-5,
    gradient_accumulation_steps=1,
    num_train_epochs=20,
    half_percision=False,
    # half_percision = compute.get_torch().cuda.is_available()
    do_evaluation=False,
    use_dropout=False)

validation_mapped_saved_path = 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum10000_do_sampleFalse_top_pNone_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'

ranker_model, tokenizer = model_loading.get_ranker_model_and_tokenizer(config)
loss_fn = F.cross_entropy

validation_generated_xsum = generated_data_loading.load_generated_dataset(validation_mapped_saved_path, 5)
validation_generated_xsum = validation_generated_xsum.select(
    range(config.num_skip, config.num_skip + config.num_examples))
validation_processed_generated_xsum = processing.convert_generated_summaries_dataset_to_regression_dataset_format(
    validation_generated_xsum, tokenizer, limit=config.num_beams, max_seq_len=512)

# do they change after a training step?
#  let's run a train step and see
# inputs = Variable(torch.randn(20, 20))
# targets = Variable(torch.randint(0, 2, (20,))).long()
# batch = [validation_processed_generated_xsum[0]['input_ids_s'], validation_processed_generated_xsum[0]['labels']]
# batch[0] = torch.stack(batch[0])
# attention = torch.stack(validation_processed_generated_xsum[0]['attention_mask_s'])


params = [np for np in ranker_model.named_parameters() if np[1].requires_grad]
# take a copy
print('num params with grad', params)
initial_params = [(name, p.clone()) for (name, p) in params]
training_args = TrainingArguments(
    output_dir="./ranker_output_dir",
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    do_train=True,
    overwrite_output_dir=False,
    # warmup_steps=0,
    fp16=config.half_percision,
    learning_rate=config.learning_rate,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    remove_unused_columns=False,
    evaluation_strategy='epoch' if config.do_evaluation else "no",
    # load_best_model_at_end=True
    dataloader_num_workers=0,
)

# train for 1 epochs
training.train_ranker(ranker_model, training_args, validation_processed_generated_xsum)

for (_, p0), (name, p1) in zip(initial_params, params):
    try:
        if True:
            assert not torch.equal(p0, p1)
            print(f'weights {name} with shape {p0.shape}: '
                  f'before update norm was {p0.norm()} and now it is {p1.norm()}.'
                  f' And the change mean absolute change per weight is {abs(p1 - p0).mean()}')
        else:
            assert torch.equal(p0, p1)
    except AssertionError:
        raise Exception(  # error message
            "{var_name} {msg}".format(
                var_name=name,
                msg='did not change!' if vars_change else 'changed!'
            )
        )

print(ranker_model.roberta.classifier.dense.bias)
