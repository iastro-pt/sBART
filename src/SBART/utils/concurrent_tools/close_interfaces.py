import numpy as np


def close_buffers(shared_buffers):
    for buffer in shared_buffers:
        buffer.close()


def kill_workers(shared_buffers, in_queue, number_workers):
    close_buffers(shared_buffers)

    print("Emptying the working queue and shutting down other workers")
    while not in_queue.empty():
        in_queue.get()  # as docs say: Remove and return an item from the queue.

    for _ in range(number_workers):
        in_queue.put(np.inf)
