def tokenize(tokenizer, texts, max_length=1024):
    tokenizer(texts,
              max_length=max_length,
              return_tensors='pt',
              padding=True,
              truncation=True)
