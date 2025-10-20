# play_file_loop_pause.py

import pyaudio
import struct
import wave
import tkinter as Tk    

# BLOCKLEN = 256
BLOCKLEN = 128

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

BLOCK_DURATION = 1000.0 * BLOCKLEN/RATE # duration in milliseconds
print('Block length: %d' % BLOCKLEN)
print('Duration of block in milliseconds: %.2f' % BLOCK_DURATION)

def fun_quit():
  global CONTINUE
  print('Good bye')
  CONTINUE = False
  root.destroy()

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

B_play = Tk.Button(root, text = 'Play', command = fun_play, width = 20)
B_pause = Tk.Button(root, text = 'Pause', command = fun_pause)
B_quit = Tk.Button(root, text = 'Quit', command = fun_quit)

# Place widgets
B_play.pack()
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

CONTINUE = True
PLAY = True

while CONTINUE:

    root.update()           # UPDATE TKINTER

    if PLAY == True:

        # Get block of samples from wave file
        input_bytes = wf.readframes(BLOCKLEN)

        # print(len(input_bytes))
        if len(input_bytes) < WIDTH * BLOCKLEN:
            wf.rewind()
            input_bytes = wf.readframes(BLOCKLEN)

        # Convert binary data to number sequence (tuple)
        signal_block = struct.unpack('h' * BLOCKLEN, input_bytes)

        # Write binary data to audio output stream
        stream.write(input_bytes, BLOCKLEN)

stream.stop_stream()
stream.close()
p.terminate()

wf.close()

print('* Finished')
