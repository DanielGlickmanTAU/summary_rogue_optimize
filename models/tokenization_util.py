from utils import compute

torch = compute.get_torch()


def tokenize(tokenizer, texts, summaries=None, max_length=None, padding=True, truncate=True):
    tokenized = tokenizer(texts, summaries,
                          max_length=max_length,
                          return_tensors='pt', padding=padding, truncation=truncate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return tokenized.to(device)
