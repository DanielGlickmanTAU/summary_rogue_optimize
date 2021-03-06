import time


def measure_time(function):
    def wrapper(*args, **kwargs):
        print(f'**** STARTING  {function.__name__} ****')
        start = time.time()
        ret = function(*args, **kwargs)
        end = time.time()
        print(f'*** {function.__name__} took {end - start} seconds ***\n')
        return ret

    return wrapper
