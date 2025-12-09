import os
from psutil import Process

def memory_usage():
    process = Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  # Memory in MB