import gc

import comet_ml
import os
import time

minimum_free_giga = 9
max_num_gpus = 1

last_write = 0


def is_university_server():
    try:
        whoami = os.popen('whoami').read()
        return 'glickman1' in whoami or 'chaimc' in whoami or 'gamir' in os.environ['HOST'] or 'rack' in os.environ[
            'HOST']
    except:
        return False


def get_cache_dir():
    if is_university_server():
        return '/home/yandex/AMNLP2021/glickman1/cache/cache'
    return None


def get_index_of_free_gpus(minimum_free_giga=minimum_free_giga):
    def get_free_gpu():
        try:
            lines = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free').readlines()
        except Exception as e:
            print('error getting free memory', e)
            return {0: 10000, 1: 10000, 2: 0, 3: 10000, 4: 0, 5: 0, 6: 0, 7: 0}

        memory_available = [int(x.split()[2]) for x in lines]
        return {index: mb for index, mb in enumerate(memory_available)}

    gpus = get_free_gpu()
    # write_gpus_to_file(gpus)
    return {index: mega for index, mega in gpus.items() if mega >= minimum_free_giga * 1000}
    # return [str(index) for index, mega in gpus.items() if mega > minimum_free_giga * 1000]


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
if is_university_server():
    os.environ['TRANSFORMERS_CACHE'] = get_cache_dir()

gpus = get_index_of_free_gpus()
gpus = list(map(str, gpus))[:max_num_gpus]
join = ','.join(gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = join
print('setting CUDA_VISIBLE_DEVICES=' + join)
if max_num_gpus == 1:
    print('working with 1 gpu:(')

import torch

old_rpr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f'{self.shape} {old_rpr(self)}'


def get_torch():
    return torch


def print_size_of_model(model):
    get_torch().save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def get_device():
    torch = get_torch()

    gpus = get_index_of_free_gpus()
    print(gpus)
    return torch.device(compute_gpu_indent(gpus) if torch.cuda.is_available() else 'cpu')


def compute_gpu_indent(gpus):
    try:
        # return 'cuda'
        best_gpu = max(gpus, key=lambda gpu_num: gpus[int(gpu_num)])
        indented_gpu_index = list(gpus.keys()).index(best_gpu)
        return 'cuda:' + str(indented_gpu_index)
    except:
        return 'cuda'


def get_device_and_set_as_global():
    d = get_device()
    get_torch().cuda.set_device(d)
    return d


def clean_memory():
    t = time.time()
    gc.collect()
    get_torch().cuda.empty_cache()
    print(f'gc took:{time.time() - t}')
