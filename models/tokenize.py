import torch


def tokenize(tokenizer, texts, max_length=1024):
    tokenized = tokenizer(texts,
                          max_length=max_length,
                          return_tensors='pt', padding="max_length", truncation=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return tokenized.to(device)
