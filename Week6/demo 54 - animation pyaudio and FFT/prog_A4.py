# plot_play_file_loop_with_animate_numpy_filter.py

import pyaudio
import wave
import matplotlib
from matplotlib import pyplot
from matplotlib import animation
import numpy as np
from scipy.signal import butter, lfilter, freqz

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

# Create Butterworth filter

ORDER = 3   # filter order
fc = 0.05
# [b, a] = butter(ORDER, 2 * fc)
[b, a] = butter(ORDER, 2 * fc, btype = 'high')
[om, H] = freqz(b, a)
f = om/(2 * np.pi) * RATE
states = np.zeros(ORDER)

# Create figure, ...

fig1 = pyplot.figure(1)
fig1.set_size_inches((6, 8))  # (width, height)

ax_x = fig1.add_subplot(3, 1, 1)
ax_H = fig1.add_subplot(3, 1, 2)
ax_y = fig1.add_subplot(3, 1, 3)

# Input signal plot
[g_x] = ax_x.plot([], [])
ax_x.set_ylim(-32000, 32000)
ax_x.set_xlim(0, 1000 * BLOCKLEN / RATE)
ax_x.set_xlabel('Time (msec)')
ax_x.set_title('Input signal')

# Frequency response plot
[g_H] = ax_H.plot(f, np.abs(H))
ax_H.set_xlim(0, RATE / 2)
ax_H.set_ylim(0, 1.2)
ax_H.set_title('Frequency response')
ax_H.set_xlabel('Frequency (Hz)')

# Output signal plot
[g_y] = ax_y.plot([], [])
ax_y.set_ylim(-32000, 32000)
ax_y.set_xlim(0, 1000 * BLOCKLEN / RATE)
ax_y.set_xlabel('Time (msec)')
ax_y.set_title('Output signal')

fig1.tight_layout()


# Define animation functions

def my_init():
    t = np.arange(BLOCKLEN) * (1000/RATE)   # time axis (milliseconds)
    g_x.set_xdata(t)
    g_y.set_xdata(t)
    return (g_x, g_y)

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

    # Update graphs
    g_x.set_ydata(x)
    g_y.set_ydata(y)

    # Convert to binary data, write to audio output stream
    stream.write(y.astype('int16').tobytes(), BLOCKLEN)

    return (g_x, g_y)


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
