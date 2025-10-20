# frequency_response_demo.py
# Plot frequency response of a filter. 

import numpy as np
from scipy.signal import butter, freqz
import matplotlib
from matplotlib import pyplot

matplotlib.use('TkAgg')
# matplotlib.use('MacOSX')    # Optional if using Mac OSX
print('The matplotlib backend is %s' % matplotlib.get_backend())  # plotting backend


# Create Butterworth filter
ORDER = 3;
fc = 0.1  												# cut-off frequency (cycles/sample)
[b, a] = butter(ORDER, 2 * fc)		# 2 * fc due to 

[om, H] = freqz(b, a)						# frequency response
f = om/(2 * np.pi)							# frequency axis

# Define figure and axes
my_fig = pyplot.figure()
ax1 = my_fig.add_subplot(2, 1, 1)		# axes
ax2 = my_fig.add_subplot(2, 1, 2)			# axes

# Set size of figure
my_fig.set_size_inches((6, 7))  # (width, height)
print('The figure height is', my_fig.get_figheight())
print('The figure width is', my_fig.get_figwidth())

# Graph of frequency response (linear frequency scale)
[g1] = ax1.plot(f, np.abs(H))
g1.set_ydata(np.abs(H))
ax1.set_xlim(0, 0.5)
ax1.set_ylim(0, 1.2)
ax1.set_xlabel('Frequency (cycles/sample)')
ax1.set_title('Frequency response (linear frequency scale)')
[g1c] = ax1.plot(fc, 1/np.sqrt(2))
g1c.set_marker('o')

# Graph of frequency response (log frequency scale)
# [g2] = ax2.semilogx(f, np.abs(H), base = 2)
[g2] = ax2.plot(f, np.abs(H))
g2.set_ydata(np.abs(H))
ax2.set_xscale('log', base = 2)
ax2.set_xlim(0.5**6, 0.5)
ax2.set_ylim(0, 1.2)
ax2.set_xlabel('Frequency (cycles/sample)')
ax2.set_title('Frequency response (log frequency scale)')
[g2c] = ax2.plot(fc, 1/np.sqrt(2))
g2c.set_marker('o')

my_fig.tight_layout()

pyplot.show()

