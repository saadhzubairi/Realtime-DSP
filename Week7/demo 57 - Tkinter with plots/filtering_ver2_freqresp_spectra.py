# filtering_ver2_freqresp_spectra.py

# Low-pass filter a signal. Vary the cut-off frequency with a slider. 


import numpy as np
from scipy.signal import butter, freqz, lfilter
import matplotlib.figure
import tkinter as Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = Tk.Tk()
root.title('Filter GUI, version 2')

cutoff_freq = Tk.DoubleVar()      # Define Tk variable
cutoff_freq.set(0.25)           # Initilize


# Create noisy signal
N = 500
n = np.arange(0, N)
x = np.sin(10 * np.pi * n/N) + 0.4 * np.random.randn(N)

# Initialize frequency response for plotting
b = [0]
a = [1]
# [b, a] = butter(2, 2 * cutoff_freq.get())
[om, H] = freqz(b, a)
f = om/(2 * np.pi)

X = np.fft.fft(x)
f_X = np.arange(len(X)) / len(x)

my_fig = matplotlib.figure.Figure()
ax_freqresp = my_fig.add_subplot(3, 1, 1)   # axes
ax_signal = my_fig.add_subplot(3, 1, 2)
ax_spectrum = my_fig.add_subplot(3, 1, 3)     # axes

[line_freqresp] = ax_freqresp.plot(f, np.abs(H), color = 'black')
ax_freqresp.set_xlim(0, 0.5)
ax_freqresp.set_ylim(0, 1.2)
ax_freqresp.set_xlabel('Frequency (cycles/sample)')

# plot dots
ax_signal.plot(n, x, 
    label = 'Noisy data', 
    # marker = '.', markersize = 5, linestyle = 'None', 
    color = 'gray') 

# plot continuous curve
[line_signal] = ax_signal.plot(n, x,
    label = 'Filtered data',
    linewidth = 2, color = 'black')

ax_signal.set_xlim(0, N)
ax_signal.set_ylim(-3, 3)
ax_signal.legend( loc = 'upper right')
ax_signal.set_xlabel('Time (index)')
ax_signal.set_title('Signals')

[x_spectrum] = ax_spectrum.plot(f_X, np.abs(X), color = 'gray', label = 'Noisy')
[y_spectrum] = ax_spectrum.plot(f_X, np.abs(X), color = 'black', label = 'Filtered')
ax_spectrum.set_xlim(0, 0.5)
ax_spectrum.set_xlabel('Frequency (cycles/sample)')
ax_spectrum.set_title('Spectra')
ax_spectrum.legend( loc = 'upper right')


# my_fig.canvas.draw()
my_fig.set_size_inches((5, 8))  # (width, height)

# Turn fig into a Tkinter widget
canvas = FigureCanvasTkAgg(my_fig, master = root)
my_fig.canvas.draw()

W1 = canvas.get_tk_widget()
W1.pack()


# Update plot when slider is moved
def updatePlot(event):
  [b, a] = butter(2, 2 * cutoff_freq.get())
  y = lfilter(b, a, x)
  [om, H] = freqz(b, a)
  
  line_signal.set_ydata(y)
  line_freqresp.set_ydata(np.abs(H))
  ax_freqresp.set_title('Frequency response (cut-off frequency = %.4f)' % cutoff_freq.get() )

  Y = np.fft.fft(y)
  y_spectrum.set_ydata(np.abs(Y))

  my_fig.canvas.draw()


def my_quit():
  global CONTINUE
  CONTINUE = False
  print('Good bye')
  root.quit()     # stops mainloop
  root.destroy()  # This prevents an error on Windows

# Define slider
S1 = Tk.Scale(root,
  length = 200, orient = 'horizontal', 
  from_ = 0.01, to = 0.49,
  resolution = 0.005,
  command = updatePlot,
  variable = cutoff_freq, label = 'Cut-off frequency')

B1 = Tk.Button(root, text = 'Quit', command = my_quit)

# Place widgets in the GUI window
S1.pack()
B1.pack()

updatePlot(None)
my_fig.tight_layout()

CONTINUE = True
while CONTINUE:
  root.update()

