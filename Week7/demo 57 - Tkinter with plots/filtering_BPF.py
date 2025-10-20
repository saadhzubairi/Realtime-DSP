# filtering_BPF.py

# Filter a signal with a bandpass filter. Vary the cut-off frequencies using sliders. 

import numpy as np
from scipy.signal import butter, freqz, lfilter, filtfilt
import matplotlib.figure
import tkinter as Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = Tk.Tk()
root.title('Filter GUI, version 2')

fc1 = Tk.DoubleVar()      # Define Tk variable
fc2 = Tk.DoubleVar()      # Define Tk variable

fc1.set(0.01)           # Initilize
fc2.set(0.10)           # Initilize

print(f"Cut-off frequencies: {fc1.get():.3f}, {fc2.get():.3f}")

# Create noisy signal
N = 500
n = np.arange(0, N)
# x = np.sin(5 * np.pi * n/N) + 0.4 * np.random.randn(N)
x = np.sin(30 * np.pi * n/N) + 0.4 * np.random.randn(N)

# Create Butterworth filter
ORDER = 4
b = [0]
a = [1] # Set filter to zero filter to iniatilize om and H
[om, H] = freqz(b, a)
f = om/(2 * np.pi)

my_fig = matplotlib.figure.Figure()
ax_freqresp = my_fig.add_subplot(2, 1, 1)   # axes
ax_signal = my_fig.add_subplot(2, 1, 2)

[line_freqresp] = ax_freqresp.plot(f, np.abs(H), 
    label = 'Frequency response', color = 'black')
ax_freqresp.set_xlim(0, 0.5)
ax_freqresp.set_ylim(0, 1.2)
ax_freqresp.legend()
ax_freqresp.set_xlabel('Frequency (cycles/sample)')

# plot dots
ax_signal.plot(n, x, 
    label = 'Noisy data', marker = '.', markersize = 5,
    linestyle = 'None', color = 'gray') 

# plot continuous curve
[line_signal] = ax_signal.plot(n, x,
    label = 'Filtered data',
    linewidth = 2, color = 'black')

ax_signal.set_xlim(0, N)
ax_signal.set_ylim(-3, 3)
ax_signal.legend()
ax_signal.set_xlabel('Time (index)')


# my_fig.canvas.draw()
my_fig.tight_layout()
my_fig.set_size_inches((6, 7))  # (width, height)

# Turn fig into a Tkinter widget
canvas = FigureCanvasTkAgg(my_fig, master = root)
my_fig.canvas.draw()

W1 = canvas.get_tk_widget()
W1.pack()

# Update plot when slider is moved
def updatePlot(event):
  global ORDER
  f1 = fc1.get()
  f2 = fc2.get()
  print([f1, f2])
  [b, a] = butter(ORDER, [2 * f1, 2 * f2], btype = 'bandpass')
  y = lfilter(b, a, x)
  [om, H] = freqz(b, a)
  
  line_signal.set_ydata(y)
  line_freqresp.set_ydata(np.abs(H))
  ax_freqresp.set_title(f"Cut-off frequencies {f1:.3f} and {f2:.3f}")
  my_fig.canvas.draw()


def my_quit():
  global CONTINUE
  CONTINUE = False
  print('Good bye')
  root.quit()     # stops mainloop
  root.destroy()  # This prevents an error on Windows

# Define sliders
S1 = Tk.Scale(root,
  length = 200, orient = 'horizontal', 
  from_ = 0.001, to = 0.499,
  resolution = 0.001,
  command = updatePlot,
  variable = fc1, label = 'Cut-off frequency 1')

S2 = Tk.Scale(root,
  length = 200, orient = 'horizontal', 
  from_ = 0.001, to = 0.499,
  resolution = 0.001,
  command = updatePlot,
  variable = fc2, label = 'Cut-off frequency 2')

B1 = Tk.Button(root, text = 'Quit', command = my_quit)

# Place widgets in the GUI window
S1.pack()
S2.pack()
B1.pack()

updatePlot(None)

CONTINUE = True
while CONTINUE:
  root.update()

