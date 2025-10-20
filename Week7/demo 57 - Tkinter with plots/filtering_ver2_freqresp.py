# filtering_ver2_freqresp.py

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
[om, H] = freqz(b, a)
f = om/(2 * np.pi)

my_fig = matplotlib.figure.Figure()
ax_freqresp = my_fig.add_subplot(2, 1, 1)   # axes
ax_signal = my_fig.add_subplot(2, 1, 2)

[line_freqresp] = ax_freqresp.plot(f, np.abs(H) , color = 'black' )
ax_freqresp.set_xlim(0, 0.5)
ax_freqresp.set_ylim(0, 1.2)
ax_freqresp.set_xlabel('Frequency (cycles/sample)')
ax_freqresp.set_title('Frequency response')

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
my_fig.set_size_inches((6, 7))  # (width, height)
my_fig.tight_layout()

# Turn fig into a Tkinter widget
canvas = FigureCanvasTkAgg(my_fig, master = root)
my_fig.canvas.draw()

W1 = canvas.get_tk_widget()
W1.pack()

# Update plot when slider is moved
def updatePlot(event):
  fc = cutoff_freq.get()
  [b, a] = butter(2, 2 * fc)
  y = lfilter(b, a, x)
  [om, H] = freqz(b, a)
  
  line_signal.set_ydata(y)
  line_freqresp.set_ydata(np.abs(H))
  ax_freqresp.set_title('Frequency response (cut-off frequency = %.4f)' % fc )
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

CONTINUE = True
while CONTINUE:
  root.update()

