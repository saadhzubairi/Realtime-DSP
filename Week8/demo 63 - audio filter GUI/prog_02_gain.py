# plot_play_file_gain.py

import pyaudio
import struct
import wave
import matplotlib.figure
from matplotlib import animation
from myfunctions import clip16

import tkinter as Tk    
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use('TkAgg')
# matplotlib.use('MacOSX')

print('The matplotlib backend is %s' % matplotlib.get_backend())             # the backend used by matplotlib

# Block length
BLOCKLEN = 256
# BLOCKLEN = 128
# BLOCKLEN = 64


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


# DEFINE TKINTER ROOT

root = Tk.Tk()    # Define root before creating figure

# Gain slider
gain = Tk.DoubleVar()       # Tk variable
gain.set(1.0)               # Initialize Tk variable
S_gain = Tk.Scale(root, label = 'Gain', variable = gain, from_ = 0, to = 2, resolution = 0.01)

# Quit button
B_quit = Tk.Button(root, text = 'Quit', command = root.quit)

# DEFINE FIGURE

fig1 = matplotlib.figure.Figure()
ax1 = fig1.add_subplot(1, 1, 1)

# Turn figure into a Tkinter widget
my_canvas = FigureCanvasTkAgg(fig1, master = root)
# fig1.canvas.draw()   # (optional ?)
C1 = my_canvas.get_tk_widget()   # canvas widget

# Place widgets
C1.pack()
S_gain.pack()
B_quit.pack(side = Tk.BOTTOM, fill = Tk.X, expand = True)

[g1] = ax1.plot([], [])


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

def my_update(i):

    # Get block of samples from wave file
    input_bytes = wf.readframes(BLOCKLEN)

    # Rewind if at end of file
    if len(input_bytes) < WIDTH * BLOCKLEN:
        wf.rewind()
        input_bytes = wf.readframes(BLOCKLEN)

    # Convert binary data to number sequence (tuple)
    signal_block = struct.unpack('h' * BLOCKLEN, input_bytes)

    A = gain.get()
    for n in range(BLOCKLEN):
        output_block[n] = int(clip16( A * signal_block[n]))

    g1.set_ydata(output_block)

    # Convert output value to binary data
    output_bytes = struct.pack('h' * BLOCKLEN, *output_block)

    # Write binary data to audio output stream
    stream.write(output_bytes, BLOCKLEN)

    return (g1,)

output_block = [0] * BLOCKLEN

my_anima = animation.FuncAnimation(
    fig1,
    my_update,
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
