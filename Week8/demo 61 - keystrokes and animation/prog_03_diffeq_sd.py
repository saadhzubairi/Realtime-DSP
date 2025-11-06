# prog_04_diffeq.py
# Play and plot a note (using a second-order difference equation)
# when the user presses a key on the keyboard.
# Uses Tkinter, Pyaudio, and Matplotlib.Animation

# import pyaudio
import sounddevice as sd
import numpy as np
import tkinter as Tk    
from scipy.signal import lfilter

import matplotlib.figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import animation

matplotlib.use('TkAgg')
# matplotlib.use('MacOSX')

print('The matplotlib backend is %s' % matplotlib.get_backend())      # Plotting backend

# BLOCKLEN: Number of frames per block 
# BLOCKLEN   = 1024
# BLOCKLEN   = 512
BLOCKLEN   = 256
# BLOCKLEN   = 128
# BLOCKLEN   = 64
# BLOCKLEN   = 32

# Question: how low can you set BLOCKLEN without causing audible drop outs? (Try different values)

WIDTH       = 2         # Bytes per sample
CHANNELS    = 1         # Mono
RATE        = 8000      # Frames per second
MAXVALUE    = 2**15-1   # Maximum allowed output signal value (because WIDTH = 2)

BLOCK_DURATION = 1000.0 * BLOCKLEN/RATE # duration in milliseconds
print('Block length: %d' % BLOCKLEN)
print('Duration of block in milliseconds: %.2f' % BLOCK_DURATION)

# Filter parameters
Ta = 1.2    # Decay time (seconds)
f1 = 350    # Frequency (Hz)

# Pole radius and angle
r = 0.01 ** ( 1.0/(Ta * RATE) )       # 0.01 for 1 percent amplitude
om1 = 2.0 * np.pi * f1/RATE

# Filter coefficients (second-order recursive filter)
a = [1, -2 * r * np.cos(om1), r ** 2]
b = [np.sin(om1)]
ORDER = 2   # filter order
states = np.zeros(ORDER)
x = np.zeros(BLOCKLEN)

# # Open the audio output stream
# p = pyaudio.PyAudio()
# PA_FORMAT = pyaudio.paInt16
# stream = p.open(
#         format      = PA_FORMAT,
#         channels    = CHANNELS,
#         rate        = RATE,
#         input       = False,
#         output      = True,
#         frames_per_buffer = BLOCKLEN)
# What if you specify a much larger value for frames_per_buffer? (try 8*BLOCKLEN)(latency)
# specify low frames_per_buffer to reduce latency


# Create an output audio stream for input and output
stream = sd.RawStream(
        samplerate = RATE,
        blocksize = BLOCKLEN,
        channels = 1,
        dtype = 'int16',
        latency = 'low')   # note this

stream.start()


KEYPRESS = False

def my_function(event):
    global KEYPRESS
    print('You pressed ' + event.char)
    if event.char == 'q':
        print('Good bye')
        root.quit()
    KEYPRESS = True

# Define Tkinter root

root = Tk.Tk()
root.bind("<Key>", my_function)

print('Press keys for sound.')
print('Press "q" to quit')

# Define figure

my_fig = matplotlib.figure.Figure()
my_ax = my_fig.add_subplot(1, 1, 1)
[g1] = my_ax.plot([], [])
my_ax.set_ylim(-32000, 32000)
my_ax.set_xlim(0, BLOCKLEN * 1000.0 / RATE)   # Time axis in milliseconds 
my_ax.set_xlabel('Time (milliseconds)')
my_ax.set_title('Signal')

my_canvas = FigureCanvasTkAgg(my_fig, master = root)    # create Tk canvas from figure
C1 = my_canvas.get_tk_widget()    # canvas widget
C1.pack()                         # place canvas widget

# Define animation functions

M1 = np.int64(BLOCKLEN/2)

def my_init():
    t = np.arange(BLOCKLEN) * 1000/RATE
    g1.set_xdata(t)
    return (g1,)

def my_update(i):
    global states
    global x
    global KEYPRESS

    if KEYPRESS:
        x[M1] = 10000.0 # Some key (not 'q') was pressed
    [y, states] = lfilter(b, a, x, zi = states)
    x[M1] = 0.0        
    KEYPRESS = False
    y = np.clip(y, -MAXVALUE, MAXVALUE)     # Clipping
    g1.set_ydata(y)                         # update plot
    stream.write(y.astype('int16').tobytes())
    # stream.write(y.astype('int16').tobytes(), BLOCKLEN)
    return (g1,)

my_anima = animation.FuncAnimation(
    my_fig,
    my_update,
    init_func = my_init,
    interval = 10,   # milliseconds (what happens if this is 200?)
    blit = True,
    cache_frame_data = False,
    repeat = False
)

Tk.mainloop()    # Start Tkinter (includes animation)

# Close audio stream
# stream.stop_stream()
stream.stop()
stream.close()
# p.terminate()

print('* Finished')
