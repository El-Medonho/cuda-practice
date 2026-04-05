import numpy as np

n = 256
k = 512
m = 128

x1 = np.random.rand(n,k)
x2 = np.random.rand(k,m)

import time

for i in range(100):
    x3 = np.matmul(x1,x2)

times = []
times2 = []

for i in range(100):
    t = time.time()
    x3 = np.matmul(x1,x2)
    t = time.time()-t
    times.append(t)

print(np.array(times).mean())