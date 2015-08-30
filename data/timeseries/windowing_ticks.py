import numpy as np
import sys

f = open(sys.argv[1],'rb')
f_out = open("ticks_windows.csv",'wb')

wMax = int(sys.argv[2])
window = np.zeros(wMax)
w = 0
while w < wMax:
    raw = f.next()
    v = float(raw[:-1])
    window[w] = v
    w += 1
f_out.write(" ".join(map(lambda x: str(x), window)) + "\n")
for raw in f:
    v = float(raw[:-1])
    window[:-1] = window[1:]
    window[-1] = v
    f_out.write(" ".join(map(lambda x: str(x), window)) + "\n")
f_out.close()

