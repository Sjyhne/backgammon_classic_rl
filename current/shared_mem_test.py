from multiprocessing import shared_memory, Process
import numpy as np

from agents import QAgent

a = np.zeros((2,), dtype=np.float16)
shm = shared_memory.SharedMemory(create=True, size=a.nbytes)

b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)

c = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)

print(b)
print(c)

def increment(lst, index, value):
    lst[index] += value


for i in range(10):
    p1 = Process(target=increment, args=[b, 0, 1])
    p2 = Process(target=increment, args=[b, 1, 2])
    p1.start()
    p2.start()
    p1.join()
    p2.join()

print(b)
print(c)
print(a)

shm.close()
shm.unlink()