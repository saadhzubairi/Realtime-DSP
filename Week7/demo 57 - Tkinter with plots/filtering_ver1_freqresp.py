# filtering_ver1_freqresp.py
# Low-pass filter a signal. Vary the cut-off frequency with a slider. 


import numpy as np
from scipy.signal import butter, lfilter, freqz
import tkinter as Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.figure

matplotlib.use('TkAgg')
# matplotlib.use('MacOSX')    # Optional if using Mac OSX
print('The matplotlib backend is %s' % matplotlib.get_backend())  # plotting backend

root = Tk.Tk()
root.title('Filter GUI, version 1')

cutoff_freq = Tk.DoubleVar()			# Define Tk variable
cutoff_freq.set(0.25)  					# Initilize

# Create noisy signal
N = 500
n = np.arange(0, N)
x = np.sin(10 * np.pi * n/N) + 0.4 * np.random.randn(N)

# Initialize frequency response for plotting
b = [0]
a = [1]
[om, H] = freqz(b, a)
f = om/(2 * np.pi)

# Define figure and plot
my_fig = matplotlib.figure.Figure()
ax_freqresp = my_fig.add_subplot(2, 1, 1)		# axes
ax_signal = my_fig.add_subplot(2, 1, 2)			# axes


print('The figure height is', my_fig.get_figheight())
print('The figure width is', my_fig.get_figwidth())

[line_signal] = ax_signal.plot(n, x)
ax_signal.set_xlim(0, N)
ax_signal.set_ylim(-3, 3)
ax_signal.set_title('Outuput signal')
ax_signal.set_xlabel('Time (index)')

[line_freqresp] = ax_freqresp.plot(f, np.abs(H))
ax_freqresp.set_xlim(0, 0.5)
ax_freqresp.set_ylim(0, 1.2)
ax_freqresp.set_xlabel('Frequency (cycles/sample)')

my_fig.set_size_inches((5, 6))  # (width, height)

# Create call-back functions

def my_quit():
	global CONTINUE
	CONTINUE = False
	print('Good bye')

# Update plot when slider is moved
def updatePlot(event):
	fc = cutoff_freq.get()
	ORDER = 4
	[b, a] = butter(ORDER, 2 * fc)
	y = lfilter(b, a, x)
	[om, H] = freqz(b, a)
	
	line_signal.set_ydata(y)
	line_freqresp.set_ydata(np.abs(H))
	ax_freqresp.set_title('Frequency response (cut-off frequency = %.4f)' % fc )
	my_fig.canvas.draw()


# Create widgets

# Define slider widget
S1 = Tk.Scale(root, label = 'Cut-off frequency',
  length = 200, orient = Tk.HORIZONTAL, from_ = 0.01, to = 0.49, resolution = 0.005,
  command = updatePlot,
  variable = cutoff_freq)

# Define canvas widget
my_canvas = FigureCanvasTkAgg(my_fig, master = root)
C1 = my_canvas.get_tk_widget()

# Define button widget
B1 = Tk.Button(root, text = 'Quit', command = my_quit)


# Place widgets in the GUI window
C1.pack()
S1.pack()			
B1.pack()

updatePlot(None)		# Run callback function

my_fig.tight_layout()

CONTINUE = True
while CONTINUE:
	root.update()

