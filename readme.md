Setup:
pip install -r requirements.txt

Running Feedback filtering:  
cd into experiments and run `python ./fewshots.py` with the following flags

Required flags:  
`--model_name_or_path` huggingface model for the generator. tested on `facebook/bart-base`  
`--dataset_name`. tested on `[xsum,cnn_dailymail]`  
`--use_gpt_dataset`. Should use GPT generated data(saved in `results_open_json.json`). accepts `True\False`  
`--ranking`. how to select generated examples. Accepts `['filter','oracle','random']`   
`--ranker_loss_fn`. Accepts `['ranking','bce']`  
`--max_train_samples`. training set size(e.g 16/32)  
`--amount_to_pass_filter`. A number between 0 and 1 indicating the portion of the generated data that is to be used for
training.  
Add the following flags as
is: `--predict_with_generate True --overwrite_output_dir True --do_train True --do_eval True --do_predict True`

Optinal flags:
`--shuffle_training_set`,`--shuffle_seed`. If you want to use a different random training set, set to True and pass some
integer as shuffle seed(we used 10,12,32)  
`--skip_first_test_eval`.default: `False`. By default, running the script will evaluate on the test set before and after
training on the self generated examples. You can use this flag to turn off the initial evaluation to save some time.

You can pass any other huggingface TrainerArguments here and they would be used for training the generator, some useful
ones are:  
`--metric_for_best_model`. used for early stopping the generator training . Accepts `loss`(default) and `rouge2`  
`--ranker_metric_for_best_model`. used for early stopping the ranker training. Accepts `loss`(default)
and `'accuracy_at_1'`  
`--learning_rate`
`--ranker_learning_rate`
`--gradient_accumulation_steps`
`ranker_gradient_accumulation_steps` default:2
`--num_train_epochs`. used for both generator and ranker training default: 15
`--evaluation_strategy` default: epoch
`--max_eval_samples` default: 256
`--per_device_train_batch_size` default: 4
`--per_device_eval_batch_size` default: 8

Some other options that we played with, but are not fully supported  
`--algorithm_cycles` default 1. Will run the whole algorithm, i.e generate -> train filter -> filter -> retrain
generator, for the amount of iterations specified. If used with `--use_gpt_dataset True`, we train on GPT data in the
first iteration and then train using self generated examples.   
`--train_filter_on`. Accepts `['train','validation','both']`. Experiments with switching the generated split which is
used for training the filter.

Feel free to message me for questions.
