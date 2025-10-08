# prog_01.py
# Read a signal from a wave file,
# and plot the signal using the animate function.

import wave
import struct
import matplotlib
from matplotlib import pyplot
from matplotlib import animation

matplotlib.use('TkAgg')
# matplotlib.use('MacOSX')

print('The matplotlib backend is %s' % pyplot.get_backend())      # Plotting backend

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

BLOCKLEN = 500    # Blocksize

BLOCK_DURATION = 1000.0 * BLOCKLEN/RATE # duration in milliseconds

print('Block length: %d' % BLOCKLEN)
print('Duration of block in milliseconds: %.1f' % BLOCK_DURATION)

# Set up plotting...

fig1 = pyplot.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)   # axes
[g1] = ax1.plot([], [])

# Define animation functions

def my_init():
    g1.set_xdata(range(BLOCKLEN))
    g1.set_ydata(BLOCKLEN * [0])
    ax1.set_ylim(-32000, 32000)
    ax1.set_xlim(0, BLOCKLEN)
    ax1.set_xlabel('Time index')
    ax1.set_title('Signal')
    return (g1,)

def my_update(i):

    # print(i)

    # Get block of samples from wave file
    input_bytes = wf.readframes(BLOCKLEN)

    # Convert binary data to sequence (tuple) of numbers
    input_block = struct.unpack('h' * BLOCKLEN, input_bytes)

    # Update plot
    g1.set_ydata(input_block)

    return (g1,)

Nplots = int(LEN/BLOCKLEN) - 1
print('Number of plots = ', Nplots)

my_anima = animation.FuncAnimation(
    fig1,
    my_update,
    frames = Nplots,
    init_func = my_init,
    interval = 50,   # milliseconds   # Try 5 and 100 and re-run
    blit = True,
    repeat = False
)
pyplot.show()   # Needed for FuncAnimation to show plots

pyplot.close()
wf.close()

print('* Finished')


