# plot_play_file_filter_numpy.py

import pyaudio
import wave
import numpy as np
import tkinter as Tk    
from scipy.signal import butter, lfilter, freqz
import matplotlib.figure
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

matplotlib.use('TkAgg')
# matplotlib.use('MacOSX')

print("Python version")
print(sys.version)

print('The matplotlib backend is %s' % matplotlib.get_backend())   # graphics backend

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
print('Duration of block in milliseconds: %.1f' % BLOCK_DURATION)

# Update filter when slider is moved
def update_filter(event):
    global a, b
    global fc
    fc = cutoff_freq.get()
    # [b, a] = butter(ORDER, 2 * fc)
    [b, a] = butter(ORDER, 2 * fc, btype = 'high')
    [om, H] = freqz(b, a)
    
    line_freqresp.set_ydata(np.abs(H))
    # fig1.canvas.draw()


# DEFINE TKINTER ROOT

root = Tk.Tk()    # Define root before creating figure

# Cut-off frequency slider
cutoff_freq = Tk.DoubleVar()            # Define Tk variable
cutoff_freq.set(0.25)                   # Initilize
S_cutoff = Tk.Scale(root, label = 'Cut-off frequency',
        variable = cutoff_freq, from_ = 0.01, to = 0.49, resolution = 0.01,
        command = update_filter)

# Quit button
B_quit = Tk.Button(root, text = 'Quit', command = root.quit)


# Create Butterworth filter
ORDER = 2   # filter order
fc = cutoff_freq.get()
[b, a] = butter(ORDER, 2 * fc, btype = 'high')
[om, H] = freqz(b, a)
f = om/(2 * np.pi)
states = np.zeros(ORDER)

# DEFINE FIGURE

fig1 = matplotlib.figure.Figure()
ax_freqresp = fig1.add_subplot(2, 1, 1)
ax_signal = fig1.add_subplot(2, 1, 2)
fig1.set_size_inches((6, 6))  # (width, height)

n = np.arange(BLOCKLEN)
output_block = np.zeros(BLOCKLEN)
# [line_signal] = ax_signal.plot(n, output_block)
[line_signal] = ax_signal.plot([], [])

ax_signal.set_ylim(-32000, 32000)
ax_signal.set_xlim(0, BLOCKLEN)
ax_signal.set_xlabel('Time (n)')
ax_signal.set_title('Output signal')

# [line_freqresp] = ax_freqresp.plot(f, np.abs(H))
[line_freqresp] = ax_freqresp.plot([], [])
ax_freqresp.set_xlim(0, 0.5)
ax_freqresp.set_ylim(0, 1.2)
ax_freqresp.set_title('Frequency response')
ax_freqresp.set_xlabel('Frequency (cycles/sample)')

fig1.tight_layout()


# Turn figure into a Tkinter widget
my_canvas = FigureCanvasTkAgg(fig1, master = root)
# fig1.canvas.draw()   # (optional ?)
C1 = my_canvas.get_tk_widget()   # canvas widget

# PLACE WIDGETS

C1.pack()
B_quit.pack(fill = Tk.X, side = Tk.BOTTOM)
S_cutoff.pack(side = Tk.TOP)

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

    line_signal.set_xdata(n)
    line_signal.set_ydata(output_block)

    update_filter(None)         # draw frequency response
    line_freqresp.set_xdata(f)

    return (line_signal, line_freqresp)

def my_update(i):

    global states

    # Get block of samples from wave file
    input_bytes = wf.readframes(BLOCKLEN)

    # Rewind if at end of file
    if len(input_bytes) < WIDTH * BLOCKLEN:
        wf.rewind()
        input_bytes = wf.readframes(BLOCKLEN)

    # Convert binary data to number sequence (tuple)
    input_block = np.frombuffer(input_bytes, dtype = 'int16')

    # filtering
    [output_block, states] = lfilter(b, a, input_block, zi = states)

    # clip before converting to 16 bit integers
    signal_block = np.clip(output_block, -MAXVALUE, MAXVALUE)   
        
    # Convert output value to binary data
    # And write binary data to audio output stream
    stream.write(output_block.astype('int16').tobytes(), BLOCKLEN)

    line_signal.set_ydata(output_block)

    return (line_signal, line_freqresp)

output_block = np.zeros(BLOCKLEN)
MAXVALUE = 2**15-1

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
