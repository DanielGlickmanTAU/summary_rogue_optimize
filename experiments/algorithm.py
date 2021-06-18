from config.argument_parsing import parse_generation_args
from data import data_loading
from models import model_loading

data_args, model_args, training_args, last_checkpoint = parse_generation_args()

model, tokenizer = model_loading.get_model_and_tokenizer(model_args)

train_dataset, eval_dataset, predict_dataset, unsupervised_data = data_loading.get_dataset(data_args, training_args,
                                                                                           tokenizer,
                                                                                           do_unsupervised=True)

print(unsupervised_data)
# generator model , generator tokenizer =

# train generator(train,validation)

# create summries and add rouge for train,validation, test

# train filter(train, validation)


# rank  unsupervised generated

# select top ranked

# insert into train set(can use some dataset feature to replace 'generated_highlight' for creating labels for generation trainig)
