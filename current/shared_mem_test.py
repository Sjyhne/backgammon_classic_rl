from multiprocessing import shared_memory, Process, Pool
import numpy as np
import multiprocessing

from agents import QAgent

a = np.zeros((4,), dtype=np.float16)
shm = shared_memory.SharedMemory(create=True, size=a.nbytes)

b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)

c = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)

l = [b, c, b, c]
idxs = range(4)
vals = [i + 1 for i in range(4)]

params = [[b, 0, 1], [c, 1, 2], [b, 2, 3], [c, 3, 4],]

print(b)
print(c)

def increment(params):
    lst = params[0]
    index = params[1]
    value = params[2]
    lst[index] += value


for i in range(10):
    with Pool(processes=4) as p:
        p.map(func=increment, iterable=params)
"""    p1 = Process(target=increment, args=[b, 0, 1])
    p2 = Process(target=increment, args=[b, 1, 2])
    p1.start()
    p2.start()
    p1.join()
    p2.join()
"""
print(b)
print(c)
print(a)

print(multiprocessing.cpu_count())

shm.close()
shm.unlink()