using padding=longest over max_length is faster(3-4 times in a very short sentence case, maybe less for real examples)

model can only get up to 1024 tokens..

when tokenizer(text,summary).. and assuming len(text)>1024, tokenizer knows to truncate from text and not summary 