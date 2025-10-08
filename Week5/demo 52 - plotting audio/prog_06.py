# prog_06.py
# Acquire the microphone signal using pyaudio,
# and plot the signal using the animate function.

import pyaudio
import struct
import matplotlib
from matplotlib import pyplot
from matplotlib import animation

WIDTH = 2           # bytes per sample
CHANNELS = 1        # mono
RATE = 8000         # frames per second
BLOCKLEN = 256     # block length in samples
BLOCKLEN = 512     # block length in samples

# How short can you make the block without having audible artifacts ?

BLOCK_DURATION = 1000.0 * BLOCKLEN/RATE # duration in milliseconds

print('Block length: %d' % BLOCKLEN)
print('Duration of block in milliseconds: %.1f' % BLOCK_DURATION)

# Open the audio output stream
p = pyaudio.PyAudio()

PA_FORMAT = p.get_format_from_width(WIDTH)
stream = p.open(
    format = PA_FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = True,
    output = False,
    frames_per_buffer = BLOCKLEN)   # optional, but can help to reduce latency

# Set up plotting...

fig1 = pyplot.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)
[g1] = ax1.plot([], [])

def my_init():
    g1.set_xdata([1000 * i / RATE for i in range(BLOCKLEN)])
    g1.set_ydata(BLOCKLEN * [0])
    ax1.set_ylim(-10000, 10000)
    ax1.set_xlim(0, BLOCK_DURATION)
    ax1.set_xlabel('Time (milliseconds)')
    ax1.set_title('Signal')
    return (g1,)

def my_update(i):

    # Read audio input stream
    signal_bytes = stream.read(BLOCKLEN, exception_on_overflow = False)
    
    # Convert binary data to number sequence (tuple)
    signal_block = struct.unpack('h' * BLOCKLEN, signal_bytes)

    g1.set_ydata(signal_block)

    return (g1,)

my_anima = animation.FuncAnimation(
    fig1,
    my_update,
    init_func = my_init,
    interval = 10,
    blit = True,
    repeat = False,
    cache_frame_data = False
)
pyplot.show()   # Needed for FuncAnimation to show plots

stream.stop_stream()
stream.close()
p.terminate()

print('* Finished')
