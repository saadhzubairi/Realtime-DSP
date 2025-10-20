# play_file_loop_filter_slider_numpy.py

# Real-time filter with cut-off frequency adjustable via a slider

import pyaudio
import wave
import numpy as np
from scipy.signal import butter, lfilter
import tkinter as Tk    

# Specify wave file
wavfile = 'author.wav'
# wavfile = 'sines.wav'
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

# Blocl length : number of frames per block
# BLOCKLEN = 1024
# BLOCKLEN = 512
# BLOCKLEN = 256
BLOCKLEN = 128
# BLOCKLEN = 64
# BLOCKLEN = 32
# BLOCKLEN = 16
# BLOCKLEN = 8
# BLOCKLEN = 4
# BLOCKLEN = 2

# How short can you make the block before you hear audible artifacts ?

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
    fc = cutoff_freq.get()
    if fc == 0:
        # cut-off frequency should not be zero
        fc = 0.02
        cutoff_freq.set(fc)
    if fc == 0.5:
        # cut-off frequency should not be 0.5
        fc = 0.48
        cutoff_freq.set(fc)

    [b, a] = butter(ORDER, 2 * fc)
    # [b, a] = butter(ORDER, 2 * fc, btype = 'high')


# DEFINE TKINTER COMPONENTS

root = Tk.Tk()    # Define root before creating figure

cutoff_freq = Tk.DoubleVar()            # Define Tk variable
cutoff_freq.set(0.15)                   # Initilize

B_quit = Tk.Button(root, text = 'Quit', command = fun_quit)
S_cutoff = Tk.Scale(root, label = 'Cut-off frequency',
        variable = cutoff_freq, from_ = 0.0, to = 0.5, resolution = 0.02,
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
    input = False,
    output = True,
    frames_per_buffer = BLOCKLEN)

# Question : what is the effect of changing frames_per_buffer?
# Try making it much larger and smaller

CONTINUE = True

while CONTINUE:

    root.update()           # UPDATE TKINTER

    # Get block of samples from wave file
    input_bytes = wf.readframes(BLOCKLEN)

    # print(len(input_bytes))
    if len(input_bytes) < WIDTH * BLOCKLEN:
        wf.rewind()
        input_bytes = wf.readframes(BLOCKLEN)

    # Convert binary data to numpy array
    input_block = np.frombuffer(input_bytes, dtype = 'int16')

    # filtering
    [y, states] = lfilter(b, a, input_block, zi = states)

    # convert output to 16 bit integers and then to binary
    stream.write(y.astype('int16').tobytes(), BLOCKLEN)

stream.stop_stream()
stream.close()
p.terminate()

wf.close()

print('* Finished')
