# prog_02.py
# Read a signal from a wave file,
# implement the amplitude modulation (AM) effect,
# and plot the signals using the animate function.

import wave
import struct
import matplotlib
from matplotlib import pyplot
from matplotlib import animation
import math

matplotlib.use('TkAgg')
# matplotlib.use('MacOSX')

print('The matplotlib backend is %s' % pyplot.get_backend())      # Plotting backend

f0 = 800    # Modulation frequency for AM effect

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

# Initialize phase for amplitude modulation effect
om = 2 * math.pi * f0 / RATE
theta = 0

# Create block (initialize to zero)
output_block = BLOCKLEN * [0]

# Set up plotting...

fig1 = pyplot.figure(1)
fig1.set_figheight(6.0) 
ax1 = fig1.add_subplot(2, 1, 1)   # axes
ax2 = fig1.add_subplot(2, 1, 2)   # axes
[g1] = ax1.plot([], [])
[g2] = ax2.plot([], [])

# Define animation functions

def my_init():
    g1.set_xdata(range(BLOCKLEN))
    g1.set_ydata(BLOCKLEN * [0])
    ax1.set_ylim(-32000, 32000)
    ax1.set_xlim(0, BLOCKLEN)
    ax1.set_xlabel('Time index')
    ax1.set_title('Input signal')

    g2.set_xdata(range(BLOCKLEN))
    g2.set_ydata(BLOCKLEN * [0])
    ax2.set_ylim(-32000, 32000)
    ax2.set_xlim(0, BLOCKLEN)
    ax2.set_xlabel('Time index')
    ax2.set_title('Output signal')

    return (g1, g2)

def my_update(i):

    global theta
    # print(i)

    # Get block of samples from wave file
    input_bytes = wf.readframes(BLOCKLEN)

    # Convert binary data to sequence (tuple) of numbers
    input_block = struct.unpack('h' * BLOCKLEN, input_bytes)

    # Go through block
    for n in range(0, BLOCKLEN):
        theta = theta + om
        output_block[n] = int( input_block[n] * math.cos(theta) )
        # output_block[n] = input_block[n]  # for no processing

    # keep theta betwen -pi and pi
    while theta > math.pi:
        theta = theta - 2*math.pi

    # Update plots
    g1.set_ydata(input_block)
    g2.set_ydata(output_block)

    return (g1, g2)

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

fig1.tight_layout()

pyplot.show()   # Needed for FuncAnimation to show plots

pyplot.close()
wf.close()

print('* Finished')

