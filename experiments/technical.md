too small learning rate caused it not to converge in reasonable time.

non deterministic logits in training because of dropout

using padding=longest over max_length is faster(3-4 times in a very short sentence case, maybe less for real examples)

generating model(bart) can only get up to 1024 tokens.. ranker(roberta) can only get up to 512

when tokenizer(text,summary).. and assuming len(text)>1024, tokenizer knows to truncate from text and not summary

how to shut down evaluation:
pass eval_strategy = 'no'(or just dont touch the parameter)

learning:
https://www.comet.ml/danielglickmantau/summary-sampling/2ed67d1a1ab34837afb248403364c70c?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step
num_examples = 100 num_beams = 1 learning_rate = 1e-5 gradient_accumulation_steps = 5

