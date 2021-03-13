import torch


def tokenize(tokenizer, texts, max_length=1024):
    tokenized = tokenizer(texts, max_length=max_length, return_tensors='pt', padding=True, truncation=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return {'input_ids': tokenized['input_ids'].to(device), 'attention_mask': tokenized['attention_mask'].to(device)}
