# plot_wave_file.py

import wave
import tkinter as Tk    
import matplotlib.figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Specify wave file
wavfile = 'author.wav'
print('Name of wave file: %s' % wavfile)

# Open wave file
wf = wave.open( wavfile, 'rb')

# Read wave file properties
RATE        = wf.getframerate()     # Frame rate (frames/second)
WIDTH       = wf.getsampwidth()     # Number of bytes per sample
LEN         = wf.getnframes()       # Signal length
CHANNELS    = wf.getnchannels()     # Number of channels

print('The file has %d channel(s).'         % CHANNELS)
print('The file has %d frames/second.'      % RATE)
print('The file has %d frames.'             % LEN)
print('The file has %d bytes per sample.'   % WIDTH)

BLOCKLEN = 1000    # Blocksize


# DEFINE TKINTER ROOT

root = Tk.Tk()    # Define root before creating figure

# Set up plotting...

fig1 = matplotlib.figure.Figure()
ax1 = fig1.add_subplot(1, 1, 1)

[g1] = ax1.plot([], [])
n = range(BLOCKLEN)
g1.set_xdata(n)
g1.set_ydata(BLOCKLEN * [0])
ax1.set_xlabel('Time (n)')

ax1.set_ylim(-32000, 32000)
ax1.set_xlim(0, BLOCKLEN)

# Turn figure into a Tkinter widget
my_canvas = FigureCanvasTkAgg(fig1, master = root)
fig1.canvas.draw()
W1 = my_canvas.get_tk_widget()

# Place widgets
W1.pack()

while True:

    root.update()           # UPDATE FIGURE

    # Get block of samples from wave file
    input_bytes = wf.readframes(BLOCKLEN)

    if len(input_bytes) < BLOCKLEN * WIDTH:
        break

    # Convert binary data to sequence (tuple) of numbers
    signal_block = np.frombuffer(input_bytes, dtype = 'int16')

    g1.set_ydata(signal_block)
    fig1.canvas.draw()                  # display the figure

wf.close()
