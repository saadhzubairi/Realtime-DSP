# demo_12.py

# demo_12 output audio    numpy   spectrum    time & freq

import pyaudio
import matplotlib
from matplotlib import pyplot
from matplotlib import animation
import numpy as np

matplotlib.use('TkAgg')
# matplotlib.use('MacOSX')

print('The matplotlib backend is %s' % pyplot.get_backend())      # Plotting backend


WIDTH = 2           # bytes per sample
CHANNELS = 1        # mono
RATE = 8000         # frames per second

# block length in samples
BLOCKLEN = 512     
BLOCKLEN = 256

print('Block length: %d' % BLOCKLEN)
print('Duration of block in milliseconds: %.1f' % (1000.0 * BLOCKLEN/RATE))

# Open the audio stream

p = pyaudio.PyAudio()
PA_FORMAT = p.get_format_from_width(WIDTH)
stream = p.open(
    format = PA_FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = True,
    output = True,
    frames_per_buffer = BLOCKLEN)   # optional, but can help to reduce latency


# Set up plots

my_fig = pyplot.figure(1)
ax1 = my_fig.add_subplot(2, 1, 1)
ax2 = my_fig.add_subplot(2, 1, 2)

t = np.arange(BLOCKLEN) * (1000/RATE)   # time axis (milliseconds)
x = np.zeros(BLOCKLEN)                  # signal
X = np.fft.rfft(x)                          # frequency spectrum of signal (real fft)
f_X = np.arange(X.size) * RATE/BLOCKLEN     # frequency axis

# signal plot
[g1] = ax1.plot([], [])                 # Create empty axis
ax1.set_xlim(0, 1000 * BLOCKLEN/RATE)   # set x-axis limits
ax1.set_ylim(-10000, 10000)             # set y-axis limits
ax1.set_xlabel('Time (msec)')
ax1.set_title('Signal')

# Frequency spectrum plot
[g2] = ax2.plot([], [])                 # Create empty axis

ax2.set_xlim(0, RATE/2)
ax2.set_ylim(0, 1000)
ax2.set_title('Frequency spectrum')
ax2.set_xlabel('Frequency (Hz)')

my_fig.tight_layout()


# Define animation functions

def my_init():
    g1.set_xdata(t)                   # x-data of plot
    g2.set_xdata(f_X)
    return (g1, g2)

def my_update(i):

    # Read audio input stream
    signal_bytes = stream.read(BLOCKLEN, exception_on_overflow = False)

    # Convert binary data to number sequence (numpy array)
    signal_block = np.frombuffer(signal_bytes, dtype = 'int16')

    # Compute frequency spectrum
    X = np.fft.rfft(signal_block) / BLOCKLEN

    g1.set_ydata(signal_block)
    g2.set_ydata(np.abs(X))

    # Convert to binary data, write to audio output stream
    stream.write(signal_block.astype('int16').tobytes(), BLOCKLEN)

    return (g1, g2)


# Read microphone, plot audio signal

my_anima = animation.FuncAnimation(
    my_fig,
    my_update,
    init_func = my_init,
    interval = 10,   # milliseconds (what happens if this is 200?)
    blit = True,
    repeat = False
)
pyplot.show()   # Needed for FuncAnimation to show plots

stream.stop_stream()
stream.close()
p.terminate()

print('* Finished')
