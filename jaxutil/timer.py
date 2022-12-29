
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