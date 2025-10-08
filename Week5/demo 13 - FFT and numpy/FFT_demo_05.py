# FFT_demo_05.py

# Real FFT

import numpy as np
from matplotlib import pyplot

N = 20
n = np.arange(0, N)     # n = [0, 1, 2, ..., N-1]
f0 = 3.0
x = np.cos(2.0 * np.pi * f0 / N * n)
X = np.fft.rfft(x)
g = np.fft.irfft(X)
err = x - g                 # reconstruction error

print('max(abs(err)) = ', np.max(np.abs(err)))

fig1 = pyplot.figure(1)

ax1 = fig1.add_subplot(2, 1, 1)   # axes
ax1.stem(n, x)
ax1.set_xlim(-1, N)
ax1.set_title('Signal')
ax1.set_xlabel('Signal index')

ax2 = fig1.add_subplot(2, 1, 2)   # axes
k = np.arange(len(X))
ax2.stem(k, np.abs(X))
ax2.set_xlim(-1, N)
ax2.set_title('Spectrum')
ax2.set_xlabel('DFT index')

fig1.tight_layout()

fig1.savefig('FFT_demo_05.pdf')
pyplot.show()

