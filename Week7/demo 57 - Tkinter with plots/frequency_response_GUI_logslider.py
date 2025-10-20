# frequency_response_GUI_logslider.py
# Plot frequency response of a filter. 
# Vary the cut-off frequency with a slider. 

# Slider controls log2(cut-off frequency),
# i.e., equal increment on log-scale

import numpy as np
from scipy.signal import butter, freqz
import tkinter as Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.figure

matplotlib.use('TkAgg')
# matplotlib.use('MacOSX')    # Optional if using Mac OSX
print('The matplotlib backend is %s' % matplotlib.get_backend())  # plotting backend

root = Tk.Tk()
root.title('Frequency Response GUI')

log2_fc = Tk.DoubleVar()		# log2( cut-off frequency ) (Tk variable)
log2_fc.set(-5)  						# Initilize

# Create Butterworth filter
ORDER = 3
fc = 2 ** log2_fc.get()
[b, a] = butter(ORDER, 2 * fc)
[om, H] = freqz(b, a)
f = om/(2 * np.pi)

# Define figure and plot
my_fig = matplotlib.figure.Figure()
ax1 = my_fig.add_subplot(2, 1, 1)		# axes
ax2 = my_fig.add_subplot(2, 1, 2)			# axes

# Set size of figure
my_fig.set_size_inches((6, 7))  # (width, height)
print('The figure height is', my_fig.get_figheight())
print('The figure width is', my_fig.get_figwidth())

[g1] = ax1.plot(f, np.abs(H))
ax1.set_xlim(0, 0.5)
ax1.set_ylim(0, 1.2)
ax1.set_xlabel('Frequency (cycles/sample)')
ax1.set_title('Frequency response')
[g1c] = ax1.plot(fc, 1/np.sqrt(2))
g1c.set_marker('o')

# [g2] = ax2.semilogx(f, np.abs(H), base = 2)
[g2] = ax2.plot(f, np.abs(H))
ax2.set_xscale('log', base = 2)
ax2.set_xlim(0.5**8, 0.5)
ax2.set_ylim(0, 1.2)
ax2.set_xlabel('Frequency (cycles/sample)')
ax2.set_title('Frequency response')
[g2c] = ax2.plot(fc, 1/np.sqrt(2))
g2c.set_marker('o')

# Create call-back functions

def my_quit():
	global CONTINUE
	CONTINUE = False
	print('Good bye')

# Update plot when slider is moved
def updatePlot(event):
	fc = 2 ** log2_fc.get()
	# print(f'The cut-off frequency is {fc:.4f} cycles/sample')
	if fc < 0.5:
		[b, a] = butter(ORDER, 2 * fc)
		[om, H] = freqz(b, a)
	
		g1.set_ydata(np.abs(H))
		ax1.set_title('Frequency response (cut-off frequency = %.3f)' % fc )
		g1c.set_xdata([fc])

		g2.set_ydata(np.abs(H))
		ax2.set_title('Frequency response (cut-off frequency = %.3f)' % fc )
		g2c.set_xdata([fc])

		my_fig.canvas.draw()
	else:
		print('The cut-off frequency must be less than 0.5')

# Define slider widget
S1 = Tk.Scale(root,
  length = 200, orient = 'horizontal', from_ = -7, to = -1.1, resolution = 0.1,
  command = updatePlot,
  label = 'log2(cut-off frequency)',
  variable = log2_fc)

# Define canvas widget
my_canvas = FigureCanvasTkAgg(my_fig, master = root)
C1 = my_canvas.get_tk_widget()

# Define button widget
B1 = Tk.Button(root, text = 'Quit', command = my_quit)

# Place widgets in the GUI window
C1.pack()		# canvas
S1.pack()		# slider
B1.pack()		# button (quit)

updatePlot(None)		# Run callback function

my_fig.tight_layout()

CONTINUE = True
while CONTINUE:
	root.update()

