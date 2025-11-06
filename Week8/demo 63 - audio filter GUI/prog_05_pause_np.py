# plot_play_file_gain_pause_numpy.py

import pyaudio
import wave
import numpy as np
import matplotlib.figure
from matplotlib import animation

import tkinter as Tk    
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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


BLOCKLEN = 256    
# BLOCKLEN = 128


def fun_pause():
  global PLAY
  # print('I am pausing')
  PLAY = False

def fun_play():
  global PLAY
  # print('I am playing')
  PLAY = True


# DEFINE TKINTER ROOT

root = Tk.Tk()    # Define root before creating figure

# Gain slider
gain = Tk.DoubleVar()       # Tk variable
gain.set(1.0)               # Initialize Tk variable
S_gain = Tk.Scale(root, label = 'Gain', variable = gain, from_ = 0, to = 2, resolution = 0.01)

# Play and pause buttons
B_play = Tk.Button(root, text = 'Play', command = fun_play)
B_pause = Tk.Button(root, text = 'Pause', command = fun_pause)

# Quit button
B_quit = Tk.Button(root, text = 'Quit', command = root.quit)


# DEFINE FIGURE

fig1 = matplotlib.figure.Figure()                     # not using pyplot
ax1 = fig1.add_subplot(1, 1, 1)
[g1] = ax1.plot([], [])

# Turn figure into a Tkinter widget
my_canvas = FigureCanvasTkAgg(fig1, master = root)
# fig1.canvas.draw()   # (optional ?)
C1 = my_canvas.get_tk_widget()   # canvas widget

# PLACE WIDGETS

C1.pack()
B_quit.pack(fill = Tk.X, side = Tk.BOTTOM)
S_gain.pack(side = Tk.TOP)

B_play.pack(side = Tk.LEFT, expand = True, fill = Tk.X)
B_pause.pack(side = Tk.RIGHT, expand = True, fill = Tk.X)


# Open the audio output stream
p = pyaudio.PyAudio()

PA_FORMAT = p.get_format_from_width(WIDTH)
stream = p.open(
    format = PA_FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = False,
    output = True,
    frames_per_buffer = BLOCKLEN)   # optional, but can help to reduce latency

# Define animation functions

def my_init():
    print('hello')
    g1.set_xdata(range(BLOCKLEN))
    ax1.set_ylim(-32000, 32000)
    ax1.set_xlim(0, BLOCKLEN)
    ax1.set_xlabel('Time (index)')
    ax1.set_title('Signal')
    return (g1,)

output_block = np.zeros(BLOCKLEN)
PLAY = True
MAXVALUE = 2**15-1

def my_update(i):

    if PLAY == True:

        # Get block of samples from wave file
        input_bytes = wf.readframes(BLOCKLEN)

        # Rewind if at end of file
        if len(input_bytes) < WIDTH * BLOCKLEN:
            wf.rewind()
            input_bytes = wf.readframes(BLOCKLEN)

        # Convert binary data to number sequence (tuple)
        signal_block = np.frombuffer(input_bytes, dtype = 'int16')
        # signal_block = struct.unpack('h' * BLOCKLEN, input_bytes)

        A = gain.get()  # Get gain from slider
        signal_block = np.clip(A * signal_block, -MAXVALUE, MAXVALUE)   
        # clips maximum value before converting to 16 bit integers

        g1.set_ydata(signal_block)
        
        # Convert output value to binary data
        # And write binary data to audio output stream
        stream.write(signal_block.astype('int16').tobytes(), BLOCKLEN)

    return (g1,)


my_anima = animation.FuncAnimation(
    fig1,
    my_update,
    # frames = Nframes, # fargs = (wf,),
    init_func = my_init,
    interval = 10,   # milliseconds (what happens if this is 200?)
    blit = True,
    cache_frame_data = False,
    repeat = False
)

Tk.mainloop()           # Start Tkinter (includes animation)

stream.stop_stream()
stream.close()
p.terminate()

wf.close()

print('* Finished')
