from models import tokenize


def summarize(model, tokenizer, texts):
    """input is list of strings batch
        output is list of strings"""
    inputs = tokenize.tokenize(tokenizer, texts)
    print('generating ', len(inputs['input_ids']), ' summaries')
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=1024)
    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for g in summary_ids]
