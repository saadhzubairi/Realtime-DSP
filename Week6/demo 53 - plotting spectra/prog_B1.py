# plot_file_using_animate_rolling.py

import struct
import wave
import matplotlib
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation

# matplotlib.use('TkAgg')
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

# Number of samples to read at a time
READ_LEN = 20

# Number of samples to plot
PLOT_LEN = 500

plot_block = PLOT_LEN * [0]

# Set up plotting...

fig1 = pyplot.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)   # axes
[g1] = ax1.plot([], [])

def my_init():
    print('Initialize')
    g1.set_xdata(range(PLOT_LEN))
    g1.set_ydata(plot_block)
    ax1.set_ylim(-32000, 32000)
    ax1.set_xlim(0, PLOT_LEN)
    ax1.set_xlabel('Time (index)')
    ax1.set_title('Signal')
    return (g1,)

def my_update(i):

    global plot_block
    
    # Get block of samples from wave file
    input_bytes = wf.readframes(READ_LEN)

    # Convert binary data to sequence (tuple) of numbers
    read_block = struct.unpack('h' * READ_LEN, input_bytes)

    # Concatenate past signal values with new signal values
    plot_block = plot_block[READ_LEN:] + list(read_block)
    
    # Update plot
    g1.set_ydata(plot_block)

    return (g1,)

Num_Plots = int(LEN/READ_LEN) - 1
print('Number of plots = ', Num_Plots)

my_anima = FuncAnimation(
    fig1,
    my_update,
    frames = Num_Plots,
    init_func = my_init,
    interval = 20,   # milliseconds
    blit = True,
    repeat = False
)
pyplot.show()   # Needed for FuncAnimation to show plots

pyplot.close()
wf.close()

print('* Finished')

