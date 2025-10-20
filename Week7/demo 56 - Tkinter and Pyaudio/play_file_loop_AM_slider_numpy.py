# play_file_loop_AM_slider_numpy.py

# Real-time filter with cut-off frequency adjustable via a slider

import pyaudio
import wave
import numpy as np
# from scipy.signal import butter, lfilter
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

# DEFINE TKINTER COMPONENTS

root = Tk.Tk()    # Define root before creating figure

AM_freq = Tk.DoubleVar()            # Define Tk variable
AM_freq.set(300)                   # Initilize

B_quit = Tk.Button(root, text = 'Quit', command = fun_quit)
S_cutoff = Tk.Scale(root, label = 'AM frequency',
        variable = AM_freq, from_ = 0.0, to = 800, resolution = 5.0)

# Place widgets
S_cutoff.pack()
B_quit.pack(fill = Tk.X)


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

CONTINUE = True

n = np.arange(BLOCKLEN)
t = n / RATE
theta = np.zeros(BLOCKLEN)

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

    theta = theta[-1] + 2 * np.pi * AM_freq.get() * t
    output_block = input_block * np.cos( theta )

    # convert output to 16 bit integers and then to binary
    stream.write(output_block.astype('int16').tobytes(), BLOCKLEN)

stream.stop_stream()
stream.close()
p.terminate()

wf.close()

print('* Finished')
