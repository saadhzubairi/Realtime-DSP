# FFT_demo_04.py
# Plot the FFT of a cosine

import numpy as np
from matplotlib import pyplot

N = 35
n = np.arange(0, N)     # n = [0, 1, 2, ..., N-1]
f0 = 2.0 				# Try also f0 = 2.5
x = np.cos(2.0 * np.pi * f0 / N * n)
X = np.fft.fft(x)

fig1 = pyplot.figure(1)

ax1 = fig1.add_subplot(2, 1, 1)   # axes
ax1.stem(n, x)
ax1.set_xlim(-1, N)
ax1.set_title('Signal')
ax1.set_xlabel('Signal index')

ax2 = fig1.add_subplot(2, 1, 2)   # axes
ax2.stem(n, np.abs(X))
# ax2.stem(n, np.angle(X))  # for phase
ax2.set_xlim(-1, N)
ax2.set_title('Spectrum')
ax2.set_xlabel('DFT index')

fig1.tight_layout()  # avoids overlapping of axis labels and axis.

pyplot.show()

fig1.savefig('FFT_demo_04.pdf')

