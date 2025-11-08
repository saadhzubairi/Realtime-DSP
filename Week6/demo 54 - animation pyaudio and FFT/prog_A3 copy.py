# plot_play_file_loop_with_animate_numpy_spectrum.py

import pyaudio
import wave
import matplotlib
from matplotlib import pyplot
from matplotlib import animation
import numpy as np
# from scipy.signal import butter, lfilter, freqz

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
fig1.set_size_inches((5, 6))  # (width, height)

ax_x = fig1.add_subplot(2, 1, 1)
ax_X = fig1.add_subplot(2, 1, 2)

x = np.zeros(BLOCKLEN)              # signal
X = np.fft.rfft(x)                  # frequency spectrum of signal (real fft)

n = np.arange(BLOCKLEN)             # Time axis
f_X = np.arange(X.size) / BLOCKLEN  # frequency axis

# signal plot
[g_x] = ax_x.plot([], [])
ax_x.set_ylim(-32000, 32000)
ax_x.set_xlim(0, BLOCKLEN)
ax_x.set_xlabel('Time (index)')
ax_x.set_title('Signal')

# Frequency spectrum plot
# [g_X] = ax_X.plot(f_X, np.abs(X))
[g_X] = ax_X.plot([], [])
ax_X.set_xlim(0, 0.5)
ax_X.set_ylim(0, 1000)
ax_X.set_title('Frequency spectrum')
ax_X.set_xlabel('Frequency (normalized)')

fig1.tight_layout()


# Define animation functions

def my_init():
    g_x.set_xdata(n)
    g_X.set_xdata(f_X)
    return (g_x, g_X)

def my_update(i):

    global states

    # Get block of samples from wave file
    signal_bytes = wf.readframes(BLOCKLEN)

    # Rewind if at end of file
    if len(signal_bytes) < WIDTH * BLOCKLEN:
        wf.rewind()
        signal_bytes = wf.readframes(BLOCKLEN)

    # Convert binary data to number sequence (numpy array)
    x = np.frombuffer(signal_bytes, dtype = 'int16')

    # Compute frequency spectrum
    X = np.fft.rfft(x) / BLOCKLEN

    # Update graphs
    g_x.set_ydata(x)
    g_X.set_ydata(np.abs(X))

    # Convert to binary data, write to audio output stream
    stream.write(signal_bytes, BLOCKLEN)

    return (g_x, g_X)


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


