import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


# top_p = 0.90
# top_p = None
# top_k = 100
# top_k = None
# num_beams = 4
# do_sample = False
# num_return_sequences = 1

@dataclass
class SearchParams:
    do_sample: bool
    top_p: Optional[int]
    top_k: Optional[int]
    num_beams: int = 4
    num_return_sequences: int = 1
    no_repeat_ngram_size: int = 3

    def _hash(self):
        return 'do_sample' + str(self.do_sample) + '_' + \
               'top_p' + str(self.top_p) + '_' + \
               'top_k' + str(self.top_k) + '_' + \
               'num_beams' + str(self.num_beams) + '_' + \
               'num_return_sequences' + str(self.num_return_sequences) + '_' + \
               'no_repeat_ngram_size' + str(self.no_repeat_ngram_size)


@dataclass
class BeamSearchParams(SearchParams):
    do_sample: bool = False
    top_p: Optional[int] = None
    top_k: Optional[int] = None


def summarize(model, tokenizer, texts, search_params: SearchParams):
    """input is list of strings batch
        output is list of strings"""
    tokenized = tokenizer(texts,
                          max_length=512,
                          return_tensors='pt', padding="max_length", truncation=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = tokenized.to(device)
    print('generating', len(inputs['input_ids']), 'summaries')
    summary_ids = model.generate(**inputs,
                                 num_beams=search_params.num_beams,
                                 do_sample=search_params.do_sample,
                                 # max_length=50,
                                 top_p=search_params.top_p,
                                 top_k=search_params.top_k,
                                 max_length=128,
                                 num_return_sequences=search_params.num_return_sequences,
                                 no_repeat_ngram_size=search_params.no_repeat_ngram_size,

                                 # early_stopping=True,
                                 )
    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for g in summary_ids]
