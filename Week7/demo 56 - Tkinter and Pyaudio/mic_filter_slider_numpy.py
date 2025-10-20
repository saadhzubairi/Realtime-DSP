# mic_filter_slider_numpy.py

# Real-time filter with cut-off frequency adjustable via a slider
# Input from microphone

import pyaudio
import numpy as np
from scipy.signal import butter, lfilter
import tkinter as Tk    


# Blocl length : number of frames per block
# BLOCKLEN = 512
# BLOCKLEN = 256
# BLOCKLEN = 128
# BLOCKLEN = 64
BLOCKLEN = 32
# BLOCKLEN = 16
# BLOCKLEN = 8
# BLOCKLEN = 4

# How short can you make the block before you hear audible artifacts ?

WIDTH = 2           # bytes per sample
CHANNELS = 1        # mono
RATE = 8000         # frames per second
DURATION = 10       # Duration in seconds

BLOCK_DURATION = 1000.0 * BLOCKLEN/RATE # duration in milliseconds

print('Block length: %d' % BLOCKLEN)
print('Duration of block in milliseconds: %.2f' % BLOCK_DURATION)


def fun_quit():
  global CONTINUE
  print('Good bye')
  CONTINUE = False

# Update filter when slider is moved
def update_filter(event):
    global a, b
    global fc
    fc = cutoff_freq.get()
    [b, a] = butter(ORDER, 2 * fc)
    # [b, a] = butter(ORDER, 2 * fc, btype = 'high')


# DEFINE TKINTER COMPONENTS

root = Tk.Tk()    # Define root before creating figure

cutoff_freq = Tk.DoubleVar()            # Define Tk variable
cutoff_freq.set(0.15)                   # Initilize

B_quit = Tk.Button(root, text = 'Quit', command = fun_quit)
S_cutoff = Tk.Scale(root, label = 'Cut-off frequency',
        variable = cutoff_freq, from_ = 0.01, to = 0.49, resolution = 0.02,
        command = update_filter)

# Place widgets
B_quit.pack(fill = Tk.X, side = Tk.BOTTOM)
S_cutoff.pack(side = Tk.TOP)

# Create Butterworth filter

ORDER = 2   # filter order
states = np.zeros(ORDER)

update_filter(None)        # Run callback function (creates filter coeffs)


# Create audio stream

p = pyaudio.PyAudio()

PA_FORMAT = p.get_format_from_width(WIDTH)
stream = p.open(
    format = PA_FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = True,
    output = True,
    frames_per_buffer = BLOCKLEN)

# Question : what is the effect of changing frames_per_buffer?
# Try making it much larger and smaller

CONTINUE = True

while CONTINUE:

    root.update()           # UPDATE FIGURE

    # Read audio input stream
    # input_bytes = stream.read(BLOCKLEN)
    input_bytes = stream.read(BLOCKLEN, exception_on_overflow = False)

    # Convert binary data to numpy array
    input_block = np.frombuffer(input_bytes, dtype = 'int16')

    # filtering
    [y, states] = lfilter(b, a, input_block, zi = states)

    # convert output to 16 bit integers and then to binary
    stream.write(y.astype('int16').tobytes(), BLOCKLEN)

stream.stop_stream()
stream.close()
p.terminate()

print('* Finished')
