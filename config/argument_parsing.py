import transformers.hf_argparser as argparser

from config.config import RankerConfig


def get_args():
    parser = argparser.HfArgumentParser(RankerConfig)
    args = parser.parse_args()
    print(args)
    return args
