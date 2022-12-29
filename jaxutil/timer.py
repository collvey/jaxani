
import time

class Timer:
    def __init__(self, description):
        self.description = description

    def __enter__(self):
        self.tic = time.perf_counter()

    def __exit__(self, *exc_info):
        toc = time.perf_counter()
        tic = self.tic
        description = self.description
        print(f"{description}: {toc - tic:0.4f} seconds")

def measure_time(func):
    # Measures the execution time of decorated function
    def wrap_func(*args, **kwargs):
        t1 = time.perf_counter()
        result = func(*args, **kwargs)
        t2 = time.perf_counter()
        print(f'{func.__name__!r}:  {(t2-t1):.4f}s')
        return result
    return wrap_func