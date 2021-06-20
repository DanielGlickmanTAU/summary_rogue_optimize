def get_checkpoint_output_dir(dataset_name, model_name, max_train_samples, learning_rate, extra=None):
    return f'./models/{dataset_name}/{max_train_samples}/{model_name}/{learning_rate}{"/" + str(extra) if extra else ""}'
