import torch


def tokenize(tokenizer, texts, max_length=512, padding=True):
    tokenized = tokenizer(texts,
                          max_length=max_length,
                          return_tensors='pt', padding=padding, truncation=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return tokenized.to(device)
