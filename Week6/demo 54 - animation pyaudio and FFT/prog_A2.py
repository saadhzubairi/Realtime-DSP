# plot_play_file_loop_with_animate_numpy.py

import pyaudio
import wave
import matplotlib
from matplotlib import pyplot
from matplotlib import animation
import numpy as np

matplotlib.use('TkAgg')
# matplotlib.use('MacOSX')

print('The matplotlib backend is %s' % pyplot.get_backend())      # Plotting backend

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
# BLOCKLEN = 64
# BLOCKLEN = 32
# BLOCKLEN = 8

BLOCK_DURATION = 1000.0 * BLOCKLEN/RATE # duration in milliseconds
print('Block length: %d' % BLOCKLEN)
print('Duration of block in milliseconds: %.2f' % BLOCK_DURATION)

# Open the audio output stream
p = pyaudio.PyAudio()

PA_FORMAT = p.get_format_from_width(WIDTH)
stream = p.open(
    format = PA_FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = False,
    output = True,
    frames_per_buffer = BLOCKLEN)

# Create figure, ...

fig1 = pyplot.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)
[g1] = ax1.plot([], [])

# Define animation functions

def my_init():
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

    # Convert binary data to number sequence (numpy array)
    signal_block = np.frombuffer(input_bytes, dtype = 'int16')

    # Update graph
    g1.set_ydata(signal_block)

    # Write binary data to audio output stream
    stream.write(input_bytes, BLOCKLEN)

    return (g1,)


my_anima = animation.FuncAnimation(
    fig1,
    my_update,
    init_func = my_init,
    interval = 10,   # milliseconds (what happens if this is 200?)
    blit = True,
    cache_frame_data = False,
    repeat = False
)
pyplot.show()   # Needed for FuncAnimation to show plots

stream.stop_stream()
stream.close()
p.terminate()

wf.close()

print('* Finished')


