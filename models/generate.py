import torch

top_p = 0.90
top_p = None
top_k = 100
top_k = None
num_beams = 4
do_sample = False
num_return_sequences = 1


def summarize(model, tokenizer, texts, do_sample, top_p, top_k, num_beams, num_return_sequences=1,no_repeat_ngram_size=3):
    """input is list of strings batch
        output is list of strings"""
    tokenized = tokenizer(texts,
                          max_length=512,
                          return_tensors='pt', padding="max_length", truncation=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = tokenized.to(device)
    print('generating', len(inputs['input_ids']), 'summaries')
    summary_ids = model.generate(**inputs,
                                 num_beams=num_beams,
                                 do_sample=do_sample,
                                 # max_length=50,
                                 top_p=top_p,
                                 top_k=top_k,
                                 max_length=128,
                                 num_return_sequences=num_return_sequences,
                                 no_repeat_ngram_size=no_repeat_ngram_size,

                                 # early_stopping=True,
                                 )
    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for g in summary_ids]
