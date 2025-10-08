# prog_05.py
# demo 52 - plotting audio
# 
# Read a signal from a wave file,
# implement the amplitude modulation (AM) effect,
# plot the signals using the animate function,
# and play the output signal using pyaudio.
# Run the process in a continuous loop.

import pyaudio
import struct
import wave
import matplotlib
from matplotlib import pyplot
from matplotlib import animation
import math

matplotlib.use('TkAgg')
# matplotlib.use('MacOSX')

print('The matplotlib backend is %s' % pyplot.get_backend())      # Plotting backend

f0 = 600    # Modulation frequency for AM effect

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

BLOCK_DURATION = 1000.0 * BLOCKLEN/RATE # duration in milliseconds
print('Block length: %d' % BLOCKLEN)
print('Duration of block in milliseconds: %.2f' % BLOCK_DURATION)

# Create block (initialize to zero)
output_block = BLOCKLEN * [0]

# Initialize phase for amplitude modulation effect
om = 2 * math.pi * f0 / RATE
theta = 0

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

# Set up plotting...

fig1 = pyplot.figure(1)
fig1.set_figwidth(8.0) 
fig1.set_figheight(6.0) 

ax1 = fig1.add_subplot(2, 1, 1)   # axes
ax2 = fig1.add_subplot(2, 1, 2)   # axes

[g1] = ax1.plot([], [])
[g2] = ax2.plot([], [])

# Define animation functions

def my_init():
    g1.set_xdata([1000 * i / RATE for i in range(BLOCKLEN)])
    g1.set_ydata(BLOCKLEN * [0])
    ax1.set_ylim(-32000, 32000)
    ax1.set_xlim(0, 1000 * BLOCKLEN / RATE)
    ax1.set_xlabel('Time (milliseconds)')
    ax1.set_title('Input Signal')

    g2.set_xdata([1000 * i / RATE for i in range(BLOCKLEN)])
    g2.set_ydata(BLOCKLEN * [0])
    ax2.set_ylim(-32000, 32000)
    ax2.set_xlim(0, 1000 * BLOCKLEN / RATE)
    ax2.set_xlabel('Time (milliseconds)')
    ax2.set_title('Output signal')

    return (g1, g2)

def my_update(i):

    global theta
    # print(i)

    # Get block of samples from wave file
    input_bytes = wf.readframes(BLOCKLEN)

    # Rewind if at end of file
    if len(input_bytes) < WIDTH * BLOCKLEN:
        wf.rewind()
        input_bytes = wf.readframes(BLOCKLEN)

    # Convert binary data to number sequence (tuple)
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

    # Convert values to binary data
    output_bytes = struct.pack('h' * BLOCKLEN, *output_block)

    # Write binary data to audio output stream
    stream.write(output_bytes, BLOCKLEN)

    return (g1, g2)

my_anima = animation.FuncAnimation(
    fig1,
    my_update,
    init_func = my_init,
    interval = 10,   # milliseconds (what happens if this is 200?)
    blit = True,
    cache_frame_data = False,
    repeat = False
)

fig1.tight_layout()
pyplot.show()   # Needed for FuncAnimation to show plots

stream.stop_stream()
stream.close()
p.terminate()

wf.close()

print('* Finished')

