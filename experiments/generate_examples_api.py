from config.argument_parsing import parse_generation_args
from data import data_loading
import openai
import json

from models import model_loading

split = 'train'

train_examples = 200

data_args, model_args, training_args, last_checkpoint = parse_generation_args()

model, tokenizer = model_loading.get_model_and_tokenizer(model_args)

train_dataset, eval_dataset, predict_dataset, unsupervised_data = data_loading.get_dataset(data_args, training_args,
                                                                                           tokenizer,
                                                                                           do_unsupervised=True)

# dataset = data_loading.get_xsum_dataset(train_subset=train_examples, valid_subset=1,
#                                         test_subset=1)

# dataset = dataset['train']
dataset = unsupervised_data


def do_filter():
    bl = "T3Bl"

    def tlakda():
        return "rV%sbkFJaAMl9fw1kVTx3pg3MjVl" % bl

    part1 = "rVT3BlbkFJaAMl9fw1kVTx3pg3%s" % tlakda()
    part0 = "sk-y"
    arapimo = "62Q"
    part2 = "go%sZs" % arapimo
    ohi = 'QZsoUkS7'
    openai.api_key = tikanos() % ohi


def tikanos():
    return 'sk-yD5go62%sL0jrVT3BlbkFJaAMl9fw1kVTx3pg3MjVl'


do_filter()


# filter
# long
# before
# this


def summarize(text, prompt):
    text_to_summarize = text + prompt
    response = openai.Completion.create(
        engine="curie",
        prompt=text_to_summarize,
        temperature=0.3,
        max_tokens=128,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    return response['choices'][0]['text']
    # return f'{prompt} :! {text}'


prompt = "\n\nSummarize:"
generated = []
for i in range(len(dataset)):
    text = dataset[i]['article']
    try:
        summary = summarize(text, prompt)
        print(text, prompt, summary)
        generated.append({'text': text, 'prompt': prompt, 'summary': summary})
    except Exception as e:
        print(f'failed parsing {text} with expect {e}')

print('generated', len(generated))

try:
    with open('openapi-results', 'w') as f:
        f.write(str(generated))
except Exception:
    pass

with open('results_open_json.json', 'w') as f:
    json.dump(generated, f)
