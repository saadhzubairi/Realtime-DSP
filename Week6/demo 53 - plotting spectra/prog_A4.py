# plot_file_with_animate_numpy_filter_spectrum.py

import wave
import matplotlib
from matplotlib import pyplot
from matplotlib import animation
import numpy as np
from scipy.signal import butter, lfilter, freqz

# matplotlib.use('TkAgg')
matplotlib.use('MacOSX')

print('The matplotlib backend is %s' % pyplot.get_backend())      # Plotting backend

# Specify wave file (uncomment one of the following)
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

BLOCKLEN = 512
# BLOCKLEN = 256
# BLOCKLEN = 128
# BLOCKLEN = 64
# BLOCKLEN = 32
# BLOCKLEN = 8

BLOCK_DURATION = 1000.0 * BLOCKLEN/RATE # duration in milliseconds
print('Block length: %d' % BLOCKLEN)
print('Duration of block in milliseconds: %.2f' % BLOCK_DURATION)


# Create Butterworth filter

ORDER = 3   # filter order
fc = 0.1
# [b, a] = butter(ORDER, 2 * fc) # low-pass filter
[b, a] = butter(ORDER, 2 * fc, btype = 'high')    # high-pass filter
[om, H] = freqz(b, a)
f_H = om/(2 * np.pi) * RATE
states = np.zeros(ORDER)

# Create figure, ...

fig1 = pyplot.figure(1)
fig1.set_size_inches((12, 7))  # (width, height)

ax_x = fig1.add_subplot(2, 2, 1)
ax_X = fig1.add_subplot(2, 2, 2)
ax_y = fig1.add_subplot(2, 2, 3)
ax_Y = fig1.add_subplot(2, 2, 4)

x = np.zeros(BLOCKLEN)               # signal
X = np.fft.rfft(x)                   # spectrum of signal (real fft)
f_X = np.arange(X.size) * RATE / BLOCKLEN   # frequency axis (Hz)

# Input signal plot
[g_x] = ax_x.plot([], [])
ax_x.set_ylim(-32000, 32000)
ax_x.set_xlim(0, 1000 * BLOCKLEN / RATE)
ax_x.set_xlabel('Time (msec)')
ax_x.set_title('Input signal')

# Input spectrum plot
[g_X] = ax_X.plot([], [])
[g_H] = ax_X.plot(f_H, 500 * np.abs(H), label = 'Frequency response (x500)', color = 'green')
ax_X.set_xlim(0, RATE/2)
ax_X.set_ylim(0, 1000)
ax_X.set_title('Spectrum of input signal')
ax_X.set_xlabel('Frequency (Hz)')
ax_X.legend()

# Output signal plot
[g_y] = ax_y.plot([], [])
ax_y.set_ylim(-32000, 32000)
ax_y.set_xlim(0, 1000 * BLOCKLEN/RATE)
ax_y.set_xlabel('Time (milliseconds)')
ax_y.set_title('Output signal')

# Output spectrum plot
[g_Y] = ax_Y.plot([], [])
ax_Y.set_xlim(0, RATE/2)
ax_Y.set_ylim(0, 1000)
ax_Y.set_title('Spectrum of output signal')
ax_Y.set_xlabel('Frequency (Hz)')

fig1.tight_layout()


# Define animation functions

def my_init():
    t = np.arange(BLOCKLEN) * (1000/RATE)   # time axis (milliseconds)
    g_x.set_xdata( t )
    g_x.set_ydata( x )
    g_y.set_xdata( t )
    g_y.set_ydata( x )
    g_X.set_xdata( f_X )
    g_X.set_ydata( np.abs(X) )
    g_Y.set_xdata(f_X)
    g_Y.set_ydata( np.abs(X) )
    return (g_x, g_y, g_X, g_Y)

def my_update(i):

    global states

    # Get block of samples from wave file
    input_bytes = wf.readframes(BLOCKLEN)

    # Rewind if at end of file
    if len(input_bytes) < WIDTH * BLOCKLEN:
        wf.rewind()
        input_bytes = wf.readframes(BLOCKLEN)

    # Convert binary data to number sequence (numpy array)
    x = np.frombuffer(input_bytes, dtype = 'int16')

    [y, states] = lfilter(b, a, x, zi = states)

    # Compute frequency spectra
    X = np.fft.rfft(x) / BLOCKLEN
    Y = np.fft.rfft(y) / BLOCKLEN

    # Update graphs
    g_x.set_ydata(x)
    g_y.set_ydata(y)
    g_X.set_ydata( np.abs(X) )
    g_Y.set_ydata( np.abs(Y) )

    return (g_x, g_y, g_X, g_Y)


Num_Plots = int(LEN/BLOCKLEN) - 1
print('Number of plots = ', Num_Plots)

my_anima = animation.FuncAnimation(
    fig1,
    my_update,
    init_func = my_init,
    frames = Num_Plots,
    # interval = ...,   # milliseconds
    # interval = BLOCK_DURATION,  # plot at real time according to signal sampling rate
    interval = 5*BLOCK_DURATION,  # plot slower than real time
    blit = True,
    repeat = False
)
pyplot.show()   # Needed for FuncAnimation to show plots

wf.close()

print('* Finished')
