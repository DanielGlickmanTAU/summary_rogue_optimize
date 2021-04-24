import utils.compute as compute

torch = compute.get_torch()
from unittest import TestCase
import models.model_loading as model_loading
import models.tokenize as tokenize
import time


class Test(TestCase):
    def test_get_ranker_model_and_tokenizer(self):
        model, tokenizer = model_loading.get_ranker_model_and_tokenizer()
        text = "Replace me by any text you'd like."
        # encoded_input = tokenizer(text, return_tensors='pt')
        start = time.time()
        encoded_input = tokenize.tokenize(tokenizer, [text], padding='max_length')
        print('time first encoding', time.time() - start)

        start = time.time()
        encoded_input = tokenize.tokenize(tokenizer, [text])
        print('time pad only longest encoding', time.time() - start)

        print(encoded_input)
        output = model(**encoded_input)
        print(output)
