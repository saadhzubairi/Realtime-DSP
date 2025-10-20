# play_file_loop_pause_gain.py

import pyaudio
import struct
import wave
import tkinter as Tk    
from myfunctions import clip16

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

# BLOCKLEN = 1024
# BLOCKLEN = 512
BLOCKLEN = 256

def fun_quit():
  global CONTINUE
  print('Good bye')
  CONTINUE = False

def fun_pause():
  global PLAY
  # print('I am pausing')
  PLAY = False

def fun_play():
  global PLAY
  # print('I am playing')
  PLAY = True

# DEFINE TKINTER COMPONENTS

root = Tk.Tk()    # Define root before creating figure
gain = Tk.DoubleVar()       # Tk variable
gain.set(1.0)               # Initialize Tk variable

B_play = Tk.Button(root, text = 'Play', command = fun_play, width = 20)
B_pause = Tk.Button(root, text = 'Pause', command = fun_pause)
B_quit = Tk.Button(root, text = 'Quit', command = fun_quit)
S_gain = Tk.Scale(root, label = 'Gain', variable = gain, from_ = 0, to = 2, resolution = 0.2)

# Place widgets
S_gain.pack()
B_play.pack(fill = Tk.X)
B_pause.pack(fill = Tk.X)
B_quit.pack(fill = Tk.X)

# Open the audio output stream
p = pyaudio.PyAudio()

PA_FORMAT = p.get_format_from_width(WIDTH)
stream = p.open(
    format = PA_FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = False,
    output = True,
    frames_per_buffer = BLOCKLEN)   # optional: frames_per_buffer helps reduce latency

output_block = BLOCKLEN * [0]

CONTINUE = True
PLAY = True

while CONTINUE:

    root.update()           # UPDATE TKINTER
    A = gain.get()

    if PLAY == True:
        # Get block of samples from wave file
        input_bytes = wf.readframes(BLOCKLEN)

        # print(len(input_bytes))
        if len(input_bytes) < WIDTH * BLOCKLEN:
            wf.rewind()
            input_bytes = wf.readframes(BLOCKLEN)

        # Convert binary data to number sequence (tuple)
        input_block = struct.unpack('h' * BLOCKLEN, input_bytes)

        for i in range(0, BLOCKLEN):
            output_block[i] = int( clip16( A * input_block[i] ) )

        # Convert output value to binary data
        output_bytes = struct.pack('h' * BLOCKLEN, *output_block)

        # Write binary data to audio output stream
        stream.write(output_bytes, BLOCKLEN)

stream.stop_stream()
stream.close()
p.terminate()

wf.close()

print('* Finished')
