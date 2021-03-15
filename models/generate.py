from models import tokenize

top_p = 0.90
top_k = 100
num_beams = 4
do_sample = True
num_return_sequences = 1


def summarize(model, tokenizer, texts):
    """input is list of strings batch
        output is list of strings"""
    inputs = tokenize.tokenize(tokenizer, texts)
    print('generating', len(inputs['input_ids']), 'summaries')
    summary_ids = model.generate(**inputs,
                                 num_beams=num_beams,
                                 do_sample=do_sample,
                                 # max_length=50,
                                 top_p=top_p,
                                 top_k=top_k,
                                 max_length=128,
                                 num_return_sequences=num_return_sequences

                                 )
    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for g in summary_ids]
