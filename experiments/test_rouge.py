from models import model_loading
from data import cnn_dataset, metrics

model, tokenizer = model_loading.get_bart_model_and_tokenizer()
cnn = cnn_dataset.get_cnn_dataset(subset=2)
rouge = metrics.get_rouge()

articles = cnn['train']['article']
highlights = cnn['train']['highlights']

inputs = tokenizer(articles,
                   max_length=1024,
                   return_tensors='pt',
                   padding=True,
                   truncation=True
                   )


print('generating')
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=1024)

decoded_summary_generated = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                             summary_ids]
print(decoded_summary_generated)