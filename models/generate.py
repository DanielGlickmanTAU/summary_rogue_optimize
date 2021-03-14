from models import tokenize


top_p = 0.92
top_k = 200
def summarize(model, tokenizer, texts):
    """input is list of strings batch
        output is list of strings"""
    inputs = tokenize.tokenize(tokenizer, texts)
    print('generating', len(inputs['input_ids']), 'summaries')
    summary_ids = model.generate(**inputs,
                                 # num_beams=4,
                                 do_sample=True,
                                 # max_length=50,
                                 top_p=top_p,
                                 top_k=top_k,
                                 max_length=200,
                                 num_return_sequences=1

                                 )
    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for g in summary_ids]
